using Arrow
using CascadeDecays
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON
using Printf
using Statistics
import ThreeBodyDecays
using ThreeBodyDecays: Recoupling, RecouplingLS, @jp_str

struct RemoveParticleTwoPhaseLS <: Recoupling
    two_ls::Tuple{Int,Int}
end

function ThreeBodyDecays.amplitude(cs::RemoveParticleTwoPhaseLS, helicities, spins)
    _, _, two_j2 = spins
    _, two_lambda2 = helicities
    exponent_num = two_j2 - two_lambda2
    iseven(exponent_num) || error("particle-2 phase requires two_j2 - two_lambda2 to be even")
    phase = isodd(div(exponent_num, 2)) ? -1 : 1
    return phase * ThreeBodyDecays.amplitude(RecouplingLS(cs.two_ls), helicities, spins)
end

# =============================================================================
# Block 1 — model inputs (no CascadeDecays types or API)
# =============================================================================

const repo_root = normpath(joinpath(@__DIR__, ".."))
const data_path = joinpath(repo_root, "data", "b-decay-events.arrow")
const params_path = joinpath(repo_root, "Analysis", "final_params_full.json")
const output_path = joinpath(repo_root, "scripts", "all_resonances_fit_fractions.txt")
const saved_reference_path = joinpath(repo_root, "notebooks", "all_resonances_fit_fractions.txt")

const nominal_mass = Dict(
    "Bp" => 5.27934,
    "D" => 1.86965,
    "K" => 0.493677,
    "D0" => 1.86483,
    "pi" => 0.13957039,
    "Dst" => 2.01026,
    "X0(2900)" => 2.866,
    "X1(2900)" => 2.904,
    "NR(0-)SPp" => 4.35,
    "NR(1.)PSp" => 4.35,
    "NR(0-)SPm" => 4.35,
    "NR(1-)PPm" => 4.35,
)

const all_resonance_names = [
    "X(3872)",
    "X(3915)(0-)",
    "chi(c2)(3930)",
    "X(3940)(1.)",
    "X(3993)",
    "Psi(4040)",
    "X(4300)",
    "NR(0-)SPp",
    "NR(1.)PSp",
    "NR(0-)SPm",
    "NR(1-)PPm",
    "X0(2900)",
    "X1(2900)",
]

const params = let
    loaded = JSON.parsefile(params_path)["value"]
    for name in ["X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)", "X(3993)", "Psi(4040)", "X(4300)", "X0(2900)", "X1(2900)"]
        key = name * "_mass"
        haskey(loaded, key) && (nominal_mass[name] = Float64(loaded[key]))
    end
    loaded
end

param_real(key) = Float64(params[key])
param_complex(key) = param_real(key * "r") * cis(param_real(key * "i"))

struct ParamCouplingSpec
    keys::Tuple{Vararg{String}}
end

resolve_coupling(spec::ParamCouplingSpec) = prod(param_complex(key) for key in spec.keys; init=1.0 + 0im)

struct SimpleVertexLS
    two_ls::NTuple{2,Int}
end

struct BWVertexLS
    two_ls::NTuple{2,Int}
    remove_particle2_phase::Bool
end

BWVertexLS(two_ls; remove_particle2_phase=false) = BWVertexLS(two_ls, remove_particle2_phase)

const VertexSpec = Union{SimpleVertexLS,BWVertexLS}

vertex_two_ls(v::SimpleVertexLS) = v.two_ls
vertex_two_ls(v::BWVertexLS) = v.two_ls

vertex_barrier_l(::SimpleVertexLS) = nothing
vertex_barrier_l(v::BWVertexLS) = div(v.two_ls[1], 2)

vertex_remove_particle2_phase(::SimpleVertexLS) = false
vertex_remove_particle2_phase(v::BWVertexLS) = v.remove_particle2_phase

struct ChainCouplingSpec
    branch::String
    propagator_two_j::Int
    root::VertexSpec
    daughter::VertexSpec
    coupling::ParamCouplingSpec
    lineshape::Symbol
end

ChainCouplingSpec(branch, propagator_two_j, root, daughter, coupling, lineshape) =
    ChainCouplingSpec(branch, propagator_two_j, root, daughter, coupling, lineshape)

function lineshape_base(lineshape::Symbol)
    name = string(lineshape)
    endswith(name, "_neg") && return Symbol(chop(name, tail=4))
    return lineshape
end

lineshape_matching_sign(lineshape::Symbol) =
    endswith(string(lineshape), "_neg") ? -1.0 + 0im : 1.0 + 0im

function bwr_decay_l(lineshape::Symbol)
    base = lineshape_base(lineshape)
    base == :bwr_l1 && return 1
    base == :bwr_l2 && return 2
    base == :x2900_bwr_l0 && return 0
    base == :x2900_bwr_l1 && return 1
    return nothing
end

bare_coupling(chain::ChainCouplingSpec) =
    resolve_coupling(chain.coupling) * lineshape_matching_sign(chain.lineshape)

struct ResonanceCouplingSpec
    name::String
    topology::Symbol
    chains::Tuple{Vararg{ChainCouplingSpec}}
end

struct ResolvedChainCouplingInfo
    spec::ChainCouplingSpec
    coupling_value::ComplexF64
end

struct CollectedResonanceCouplingInfo
    spec::ResonanceCouplingSpec
    chains::Tuple{Vararg{ResolvedChainCouplingInfo}}
end

function production_coupling_key(name::String)
    name == "X(3940)(1.)" &&
        return "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0"
    return "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
end

function resonance_coupling_spec(name::String)
    if name == "X(3872)"
        total = ParamCouplingSpec(("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0",))
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec("l0", 2, BWVertexLS((2, 2)), SimpleVertexLS((0, 2)), total, :bwr_ls_l0_below_threshold_neg),
                ChainCouplingSpec(
                    "l2", 2, BWVertexLS((2, 2)), SimpleVertexLS((4, 2)),
                    ParamCouplingSpec(("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0", "X(3872)->Dst.D_g_ls_1")),
                    :bwr_ls_l2_below_threshold_neg,
                ),
            ),
        )
    elseif name == "X(3915)(0-)"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 0,
                    BWVertexLS((0, 0)), BWVertexLS((2, 2)),
                    ParamCouplingSpec(("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0",)),
                    :bwr_l1_neg,
                ),
            ),
        )
    elseif name == "chi(c2)(3930)"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 4,
                    BWVertexLS((4, 4)), BWVertexLS((4, 2)),
                    ParamCouplingSpec(("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0",)),
                    :bwr_l2_neg,
                ),
            ),
        )
    elseif name == "X(3940)(1.)" || name == "X(3993)" || name == "X(4300)"
        l0_lineshape = name == "X(3993)" ? :bwr_ls_l0_neg : :bwr_ls_l0
        l2_lineshape = name == "X(3993)" ? :bwr_ls_l2_neg : :bwr_ls_l2
        total = ParamCouplingSpec((production_coupling_key(name),))
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec("l0", 2, BWVertexLS((2, 2)), SimpleVertexLS((0, 2)), total, l0_lineshape),
                ChainCouplingSpec(
                    "l2", 2, BWVertexLS((2, 2)), SimpleVertexLS((4, 2)),
                    ParamCouplingSpec((production_coupling_key(name), "$(name)->Dst.D_g_ls_1")),
                    l2_lineshape,
                ),
            ),
        )
    elseif name == "Psi(4040)"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 2,
                    BWVertexLS((2, 2)), BWVertexLS((2, 2)),
                    ParamCouplingSpec(("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0",)),
                    :bwr_l1,
                ),
            ),
        )
    elseif name == "NR(0-)SPp"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 0, SimpleVertexLS((0, 0)), BWVertexLS((2, 2)),
                    ParamCouplingSpec(("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0",)),
                    :nr_exp,
                ),
            ),
        )
    elseif name == "NR(1.)PSp"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 2, BWVertexLS((2, 2)), SimpleVertexLS((0, 2)),
                    ParamCouplingSpec(("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0",)),
                    :constant_neg,
                ),
            ),
        )
    elseif name == "NR(0-)SPm"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 0, SimpleVertexLS((0, 0)), BWVertexLS((2, 2)),
                    ParamCouplingSpec(("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0",)),
                    :constant,
                ),
            ),
        )
    elseif name == "NR(1-)PPm"
        return ResonanceCouplingSpec(
            name, :standard,
            (
                ChainCouplingSpec(
                    "default", 2,
                    BWVertexLS((2, 2)), BWVertexLS((2, 2)),
                    ParamCouplingSpec(("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0",)),
                    :constant,
                ),
            ),
        )
    elseif name == "X0(2900)"
        return ResonanceCouplingSpec(
            name, :dk,
            (
                ChainCouplingSpec(
                    "default", 0,
                    BWVertexLS((2, 2)), BWVertexLS((0, 0)),
                    ParamCouplingSpec(("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0",)),
                    :x2900_bwr_l0,
                ),
            ),
        )
    elseif name == "X1(2900)"
        total = ParamCouplingSpec(("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0",))
        daughter = BWVertexLS((2, 0))
        return ResonanceCouplingSpec(
            name, :dk,
            (
                ChainCouplingSpec("l0", 2, BWVertexLS((0, 0); remove_particle2_phase=true), daughter, total, :x2900_bwr_l1),
                ChainCouplingSpec(
                    "l1", 2, BWVertexLS((2, 2); remove_particle2_phase=true), daughter,
                    ParamCouplingSpec(("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0", "Bp->X1(2900).Dst_g_ls_1")),
                    :x2900_bwr_l1,
                ),
                ChainCouplingSpec(
                    "l2", 2, BWVertexLS((4, 4); remove_particle2_phase=true), daughter,
                    ParamCouplingSpec(("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0", "Bp->X1(2900).Dst_g_ls_2")),
                    :x2900_bwr_l1,
                ),
            ),
        )
    end
    error("No coupling spec implemented for $(name).")
end

function collect_resonance_coupling_info(name::String)
    spec = resonance_coupling_spec(name)
    chains = ntuple(i -> begin
            chain = spec.chains[i]
            ResolvedChainCouplingInfo(chain, resolve_coupling(chain.coupling))
        end, length(spec.chains))
    return CollectedResonanceCouplingInfo(spec, chains)
end

function lineshape_param_keys(resonance_name::String, chain::ChainCouplingSpec)
    keys = String[]
    base = lineshape_base(chain.lineshape)
    if base in (:bwr_l1, :bwr_l2) ||
       base in (:bwr_ls_l0, :bwr_ls_l2, :bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
        push!(keys, resonance_name * "_width")
        base != :bwr_l1 && base != :bwr_l2 && push!(keys, resonance_name * "_theta0")
    elseif base == :nr_exp
        append!(keys, ["NR(0-)SPp_alpha", "NR(0-)SPp_beta"])
    elseif base in (:x2900_bwr_l0, :x2900_bwr_l1)
        push!(keys, resonance_name * "_width")
    end
    return keys
end

function parametrization_keys(resonance_name::String, chain::ChainCouplingSpec)
    return unique(vcat(collect(chain.coupling.keys), lineshape_param_keys(resonance_name, chain)))
end

function build_resonance_inputs_dataframe()
    rows = NamedTuple[]
    for name in all_resonance_names
        spec = resonance_coupling_spec(name)
        for chain in spec.chains
            bare = bare_coupling(chain)
            push!(rows, (
                resonance_name=name,
                branch=chain.branch,
                topology=String(spec.topology),
                nominal_mass=nominal_mass[name],
                propagator_two_j=chain.propagator_two_j,
                root_two_ls=vertex_two_ls(chain.root),
                root_l=vertex_barrier_l(chain.root),
                root_remove_particle2_phase=vertex_remove_particle2_phase(chain.root),
                daughter_two_ls=vertex_two_ls(chain.daughter),
                daughter_l=vertex_barrier_l(chain.daughter),
                coupling_param_keys=join(chain.coupling.keys, ";"),
                bare_coupling_re=real(bare),
                bare_coupling_im=imag(bare),
                lineshape=String(chain.lineshape),
                bwr_l=bwr_decay_l(chain.lineshape),
                parametrization=join(parametrization_keys(name, chain), ";"),
            ))
        end
    end
    return DataFrame(rows)
end

const resonance_inputs_df = build_resonance_inputs_dataframe()
const resonance_coupling_info = Dict(
    name => collect_resonance_coupling_info(name) for name in all_resonance_names
)

# =============================================================================
# Block 2 — CascadeDecays construction
# (lineshape builders need event context; combined with block 3 in the event loop)
# =============================================================================

const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))
const kinematic_task = KinematicTask((topology, dk_topology))

breakup_momentum(m0, m1, m2) =
    sqrt(complex(((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)))

nominal_vertex_matching_factor(l, d, m0, m1, m2) = 1 / BlattWeisskopf{l}(d)(m0^2, m1^2, m2^2)

function event_context(row)
    pDminus = FourVector(row.Dm_px, row.Dm_py, row.Dm_pz; E=row.Dm_E)
    pD0 = FourVector(row.D0_px, row.D0_py, row.D0_pz; E=row.D0_E)
    pKplus = FourVector(row.Kp_px, row.Kp_py, row.Kp_pz; E=row.Kp_E)
    piplus = FourVector(row.pip_px, row.pip_py, row.pip_pz; E=row.pip_E)
    objs = (pD0, piplus, pDminus, pKplus)
    P_Dst = pD0 + piplus
    P_R = P_Dst + pDminus
    P_B = P_R + pKplus
    masses = SystemMasses(mass.(objs)...; m0=mass(P_B))
    system = CascadeSystem(external_spins, masses)
    point = KinematicPoint(kinematic_task, objs)
    return (
        pDminus=pDminus,
        pD0=pD0,
        pKplus=pKplus,
        piplus=piplus,
        P_Dst=P_Dst,
        P_R=P_R,
        P_B=P_B,
        system=system,
        point=point,
    )
end

bwr_lineshape(ctx, m0, width, l) = begin
    q0 = real(breakup_momentum(m0, nominal_mass["Dst"], nominal_mass["D"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = m0 * width / (2q0) * m0 / ff(q0)^2
    MultichannelBreitWigner(m0, [(; gsq, ma=nominal_mass["Dst"], mb=nominal_mass["D"], l, d=3.0)])
end

function ad_hoc_mass(m0, m_min, m_max)
    k = (m_max - m_min) / 2
    return k * (1 + tanh((2m0 - (m_max + m_min)) / k / 4)) + m_min
end

function bwr_ls_q0(name::String; below_threshold=false)
    m0 = nominal_mass[name]
    q0_mass = below_threshold ?
              ad_hoc_mass(m0, nominal_mass["Dst"] + nominal_mass["D"], nominal_mass["Bp"] - nominal_mass["K"]) :
              m0
    return real(breakup_momentum(q0_mass, nominal_mass["Dst"], nominal_mass["D"]))
end

function bwr_ls_coupling_params(name::String)
    theta0 = param_real(name * "_theta0")
    return (; gamma0=cos(theta0), gamma2=sin(theta0))
end

function build_bwr_ls_lineshape(ctx, name::String; below_threshold=false)
    (; gamma0, gamma2) = bwr_ls_coupling_params(name)
    q0 = bwr_ls_q0(name; below_threshold)
    ff0 = BlattWeisskopf{0}(3.0)
    ff2 = BlattWeisskopf{2}(3.0)
    channels = [
        (; gsq=param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma0^2 / ff0(q0)^2,
            ma=nominal_mass["Dst"], mb=nominal_mass["D"], l=0, d=3.0),
        (; gsq=param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma2^2 / ff2(q0)^2,
            ma=nominal_mass["Dst"], mb=nominal_mass["D"], l=2, d=3.0),
    ]
    return MultichannelBreitWigner(nominal_mass[name], channels)
end

function chain_lineshape_matching_factor(ctx, resonance_name::String, chain::ChainCouplingSpec)
    sign = lineshape_matching_sign(chain.lineshape)
    base = lineshape_base(chain.lineshape)
    if base in (:bwr_ls_l0, :bwr_ls_l0_below_threshold)
        return sign * bwr_ls_coupling_params(resonance_name).gamma0
    elseif base in (:bwr_ls_l2, :bwr_ls_l2_below_threshold)
        below_threshold = base == :bwr_ls_l2_below_threshold
        gamma2 = bwr_ls_coupling_params(resonance_name).gamma2
        q0 = bwr_ls_q0(resonance_name; below_threshold)
        return sign * gamma2 / BlattWeisskopf{2}(3.0)(q0)
    elseif base in (:bwr_l1, :bwr_l2, :constant)
        return sign
    end
    return 1.0
end

function chain_lineshape_dynamic_matching_factor(_ctx, _resonance_name::String, chain::ChainCouplingSpec)
    base = lineshape_base(chain.lineshape)
    base in (:bwr_ls_l2, :bwr_ls_l2_below_threshold) || return 1.0
    breakup_from_sigma = sigma -> breakup_momentum(sqrt(sigma), nominal_mass["Dst"], nominal_mass["D"])
    return BlattWeisskopf{2}(3.0)(breakup_from_sigma)
end

function x2900_bwr_lineshape(ctx, name, l)
    q0 = real(breakup_momentum(nominal_mass[name], nominal_mass["D"], nominal_mass["K"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = nominal_mass[name] * param_real(name * "_width") / (2q0) * nominal_mass[name] / ff(q0)^2
    return MultichannelBreitWigner(
        nominal_mass[name],
        [(; gsq, ma=nominal_mass["D"], mb=nominal_mass["K"], l, d=3.0)],
    )
end

function build_chain_lineshape(ctx, resonance_name, chain::ChainCouplingSpec)
    base = lineshape_base(chain.lineshape)
    if base in (:bwr_ls_l0, :bwr_ls_l2, :bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
        below_threshold = base in (:bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
        return build_bwr_ls_lineshape(ctx, resonance_name; below_threshold)
    elseif base in (:bwr_l1, :bwr_l2)
        return bwr_lineshape(
            ctx, nominal_mass[resonance_name], param_real(resonance_name * "_width"),
            bwr_decay_l(chain.lineshape),
        )
    elseif base == :constant
        return ConstantLineshape(1.0 + 0.0im)
    elseif base == :nr_exp
        alpha = param_real("NR(0-)SPp_alpha")
        beta = param_real("NR(0-)SPp_beta")
        nr_factor = -exp(-(alpha + 1im * beta) * (mass(ctx.P_R)^2 - nominal_mass["NR(0-)SPp"]^2))
        return ConstantLineshape(nr_factor)
    elseif base in (:x2900_bwr_l0, :x2900_bwr_l1)
        return x2900_bwr_lineshape(ctx, resonance_name, bwr_decay_l(chain.lineshape))
    end
    error("Unknown lineshape $(chain.lineshape) for $(resonance_name).")
end

function _propagator_spin_norm(chain)
    # v0.1.0 includes prod(sqrt(two_j + 1)) in amplitude; divide to keep TF-PWA-aligned yields.
    return prod(sqrt(two_j + 1) for two_j in chain.propagator_two_js; init=1.0)
end

function _root_vertex(root_two_ls, root_l=nothing; remove_root_particle2_phase=false)
    recoupling = remove_root_particle2_phase ? RemoveParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    return root_l === nothing ? Vertex(recoupling) : Vertex(recoupling, BlattWeisskopf{root_l}(3.0))
end

function _decay_vertex(decay_two_ls, decay_l=nothing)
    return decay_l === nothing ?
           Vertex(RecouplingLS(decay_two_ls)) :
           Vertex(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(3.0))
end

function build_standard_chain(lineshape, two_j, root_two_ls, decay_two_ls; root_l=nothing, decay_l=nothing)
    return DecayChain(
        topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => Propagator(two_j, lineshape),
        ),
        vertices=(
            (((1, 2), 3), 4) => _root_vertex(root_two_ls, root_l),
            ((1, 2), 3) => _decay_vertex(decay_two_ls, decay_l),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function build_dk_chain(lineshape, two_j, root_two_ls, dk_two_ls; root_l=nothing, dk_l=nothing, remove_root_particle2_phase=false)
    return DecayChain(
        dk_topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => Propagator(two_j, lineshape),
        ),
        vertices=(
            ((1, 2), (3, 4)) => _root_vertex(root_two_ls, root_l; remove_root_particle2_phase),
            (3, 4) => _decay_vertex(dk_two_ls, dk_l),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function standard_vertex_matching_factor(resonance_name; root_l=nothing, decay_l=nothing)
    m_r = nominal_mass[resonance_name]
    root = root_l === nothing ? 1.0 :
           nominal_vertex_matching_factor(root_l, 3.0, nominal_mass["Bp"], m_r, nominal_mass["K"])
    decay = decay_l === nothing ? 1.0 :
            nominal_vertex_matching_factor(decay_l, 3.0, m_r, nominal_mass["Dst"], nominal_mass["D"])
    return root * decay
end

function dk_vertex_matching_factor(resonance_name; root_l=nothing, dk_l=nothing)
    m_r = nominal_mass[resonance_name]
    root = root_l === nothing ? 1.0 :
           nominal_vertex_matching_factor(root_l, 3.0, nominal_mass["Bp"], m_r, nominal_mass["Dst"])
    dk = dk_l === nothing ? 1.0 :
         nominal_vertex_matching_factor(dk_l, 3.0, m_r, nominal_mass["D"], nominal_mass["K"])
    return root * dk
end

function chain_vertex_matching_factor(resonance_name, topology::Symbol, chain::ChainCouplingSpec)
    root_l = vertex_barrier_l(chain.root)
    daughter_l = vertex_barrier_l(chain.daughter)
    if topology == :standard
        return standard_vertex_matching_factor(resonance_name; root_l=root_l, decay_l=daughter_l)
    elseif topology == :dk
        return dk_vertex_matching_factor(resonance_name; root_l=root_l, dk_l=daughter_l)
    end
    error("Unknown topology $(topology).")
end

function build_chain_from_spec(ctx, info::CollectedResonanceCouplingInfo, chain_info::ResolvedChainCouplingInfo)
    chain = chain_info.spec
    resonance_name = info.spec.name
    lineshape = build_chain_lineshape(ctx, resonance_name, chain)
    dynamic_factor = chain_lineshape_dynamic_matching_factor(ctx, resonance_name, chain)
    full_lineshape = dynamic_factor === 1.0 ? lineshape : dynamic_factor * lineshape
    if info.spec.topology == :dk
        return build_dk_chain(
            full_lineshape,
            chain.propagator_two_j,
            vertex_two_ls(chain.root),
            vertex_two_ls(chain.daughter);
            root_l=vertex_barrier_l(chain.root),
            dk_l=vertex_barrier_l(chain.daughter),
            remove_root_particle2_phase=vertex_remove_particle2_phase(chain.root),
        )
    end
    return build_standard_chain(
        full_lineshape,
        chain.propagator_two_j,
        vertex_two_ls(chain.root),
        vertex_two_ls(chain.daughter);
        root_l=vertex_barrier_l(chain.root),
        decay_l=vertex_barrier_l(chain.daughter),
    )
end

function chain_matching_factor(info::CollectedResonanceCouplingInfo, ctx, chain_idx::Int)
    chain = info.chains[chain_idx].spec
    return chain_vertex_matching_factor(info.spec.name, info.spec.topology, chain) *
           chain_lineshape_matching_factor(ctx, info.spec.name, chain)
end

function chain_matching_factors(info::CollectedResonanceCouplingInfo, ctx)
    return ntuple(i -> chain_matching_factor(info, ctx, i), length(info.chains))
end

function build_resonance_cascade(info::CollectedResonanceCouplingInfo, ctx)
    chains = ntuple(i -> build_chain_from_spec(ctx, info, info.chains[i]), length(info.chains))
    couplings = ntuple(i -> info.chains[i].coupling_value, length(info.chains))
    matching_factors = chain_matching_factors(info, ctx)
    effective_couplings = ntuple(
        i -> couplings[i] * matching_factors[i] / _propagator_spin_norm(chains[i]),
        length(chains),
    )
    branch_names = ntuple(i -> info.chains[i].spec.branch, length(info.chains))
    return CascadeDecay(
        chains,
        ctx.system,
        topology;
        couplings=effective_couplings,
        names=branch_names,
    )
end

# =============================================================================
# Block 3 — evaluate CascadeDecay on a kinematic point
# =============================================================================

# All external spins are 0, so amplitude(cascade, point) returns a 1-element helicity array.
evaluate_cascade_amplitude(cascade::CascadeDecay, point::KinematicPoint) =
    only(amplitude(cascade, point))

function selected_cd_amplitude(ctx, name::String)
    cascade = build_resonance_cascade(resonance_coupling_info[name], ctx)
    return evaluate_cascade_amplitude(cascade, ctx.point)
end

# =============================================================================
# Fit-fraction analysis
# =============================================================================

function compute_fit_fractions(component_names, component_amps_by_event, weights)
    length(weights) == length(component_amps_by_event) ||
        error("weights length ($(length(weights))) must match events ($(length(component_amps_by_event)))")
    coherent_norm = sum(weights[idx] * abs2(sum(component_amps)) for (idx, component_amps) in pairs(component_amps_by_event))
    component_norms = [
        sum(weights[idx] * abs2(component_amps[i]) for (idx, component_amps) in pairs(component_amps_by_event))
        for i in eachindex(component_names)
    ]
    fit_fractions = component_norms ./ coherent_norm
    incoherent_norm = sum(component_norms)
    interference_fraction = (coherent_norm - incoherent_norm) / coherent_norm
    return fit_fractions, interference_fraction, component_norms, coherent_norm, sum(weights)
end

function save_fit_fractions(path::String, component_names, fit_fractions, interference_fraction, component_norms)
    open(path, "w") do io
        println(io, join(["component", "sum_weight_times_abs2", "fit_fraction", "fit_fraction_percent"], '\t'))
        for (name, norm, fraction) in zip(component_names, component_norms, fit_fractions)
            println(io, join(string.((name, norm, fraction, 100fraction)), '\t'))
        end
        println(io, join(string.(("interference", "", interference_fraction, 100interference_fraction)), '\t'))
    end
end

function load_saved_fit_fractions(path::String)
    isfile(path) || return nothing
    saved = Dict{String,Float64}()
    for line in eachline(path)
        startswith(line, "component") && continue
        fields = split(line, '\t')
        length(fields) >= 3 || continue
        saved[fields[1]] = parse(Float64, fields[3])
    end
    return saved
end

function main()
    println("All-resonance fit fractions from b-decay-events.arrow")
    println("====================================================")
    println("Input:  ", data_path)
    println("Output: ", output_path)
    println("Model input rows (one per chain branch): ", nrow(resonance_inputs_df))
    println()

    df = DataFrame(Arrow.Table(data_path))
    n_events = nrow(df)
    weights = df.weight
    println("Loaded events: ", n_events)
    println("Using event weights from column: weight")
    println(@sprintf("  sum(weight) = %.6e", sum(weights)))
    println("Computing per-resonance amplitudes...")

    resonance_amps_by_event = Vector{Vector{ComplexF64}}()
    sizehint!(resonance_amps_by_event, n_events)

    for idx in 1:n_events
        ctx = event_context(df[idx, :])
        push!(
            resonance_amps_by_event,
            ComplexF64[selected_cd_amplitude(ctx, name) for name in all_resonance_names],
        )
        idx % 10_000 == 0 && println("  processed ", idx, " / ", n_events)
    end

    fit_fractions, interference_fraction, component_norms, coherent_norm, _ =
        compute_fit_fractions(all_resonance_names, resonance_amps_by_event, weights)
    save_fit_fractions(output_path, all_resonance_names, fit_fractions, interference_fraction, component_norms)

    println()
    println("CascadeDecays weighted fit fractions (sum w|A|^2 / sum w|A_total|^2):")
    for (name, fraction) in zip(all_resonance_names, fit_fractions)
        println(@sprintf("  %-16s %.6e  (%.4f%%)", name, fraction, 100fraction))
    end
    println(@sprintf("  %-16s %.6e  (%.4f%%)", "interference", interference_fraction, 100interference_fraction))
    println()
    println("Saved: ", output_path)

    saved = load_saved_fit_fractions(saved_reference_path)
    if saved === nothing
        println()
        println("No saved reference found at: ", saved_reference_path)
        return fit_fractions, interference_fraction
    end

    println()
    println("Comparison to saved reference (", basename(saved_reference_path), "):")
    println(@sprintf("  %-16s %12s %12s %12s", "component", "computed", "saved", "delta"))
    max_abs_delta = 0.0
    for name in all_resonance_names
        saved_val = get(saved, name, NaN)
        delta = fit_fractions[findfirst(==(name), all_resonance_names)] - saved_val
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", name, fit_fractions[findfirst(==(name), all_resonance_names)], saved_val, delta))
    end
    if haskey(saved, "interference")
        delta = interference_fraction - saved["interference"]
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", "interference", interference_fraction, saved["interference"], delta))
    end
    println(@sprintf("Max |delta| = %.6e", max_abs_delta))
    return fit_fractions, interference_fraction
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
