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

resolve_coupling_keys(keys) = prod(param_complex(key) for key in keys; init=1.0 + 0im)

function push_chain!(
    rows,
    name::String,
    branch::String,
    topology::Symbol;
    propagator_two_j::Int,
    root_two_ls::NTuple{2,Int},
    daughter_two_ls::NTuple{2,Int},
    coupling_keys,
    lineshape::Symbol,
    root_remove_particle2_phase::Bool=false,
)
    push!(rows, (
        resonance_name=name,
        branch=branch,
        topology=String(topology),
        nominal_mass=nominal_mass[name],
        propagator_two_j=propagator_two_j,
        root_two_l=root_two_ls[1],
        root_two_s=root_two_ls[2],
        root_l=div(root_two_ls[1], 2),
        root_remove_particle2_phase=root_remove_particle2_phase,
        daughter_two_l=daughter_two_ls[1],
        daughter_two_s=daughter_two_ls[2],
        daughter_l=div(daughter_two_ls[1], 2),
        coupling_keys=coupling_keys,
        lineshape=String(lineshape),
    ))
end

function lineshape_base(lineshape)
    name = lineshape isa Symbol ? string(lineshape) : lineshape
    endswith(name, "_neg") && return Symbol(chop(name, tail=4))
    return lineshape isa Symbol ? lineshape : Symbol(name)
end

lineshape_matching_sign(lineshape) =
    endswith(string(lineshape), "_neg") ? -1.0 + 0im : 1.0 + 0im

function bwr_decay_l(lineshape)
    base = lineshape_base(lineshape)
    base == :bwr_l1 && return 1
    base == :bwr_l2 && return 2
    base == :x2900_bwr_l0 && return 0
    base == :x2900_bwr_l1 && return 1
    return nothing
end

function production_coupling_key(name::String)
    name == "X(3940)(1.)" &&
        return "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0"
    return "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
end

function build_resonance_chains_df()
    rows = NamedTuple[]
    total_x3872 = ("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0",)
    push_chain!(rows, "X(3872)", "l0", :DxD;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        coupling_keys=total_x3872, lineshape=:bwr_ls_l0_below_threshold_neg)
    push_chain!(rows, "X(3872)", "l2", :DxD;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
        coupling_keys=(total_x3872..., "X(3872)->Dst.D_g_ls_1"),
        lineshape=:bwr_ls_l2_below_threshold_neg)
    push_chain!(rows, "X(3915)(0-)", "default", :DxD;
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1_neg)
    push_chain!(rows, "chi(c2)(3930)", "default", :DxD;
        propagator_two_j=4, root_two_ls=(4, 4), daughter_two_ls=(4, 2),
        coupling_keys=("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l2_neg)
    for name in ("X(3940)(1.)", "X(3993)", "X(4300)")
        l0_lineshape = name == "X(3993)" ? :bwr_ls_l0_neg : :bwr_ls_l0
        l2_lineshape = name == "X(3993)" ? :bwr_ls_l2_neg : :bwr_ls_l2
        total = (production_coupling_key(name),)
        push_chain!(rows, name, "l0", :DxD;
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
            coupling_keys=total, lineshape=l0_lineshape)
        push_chain!(rows, name, "l2", :DxD;
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
            coupling_keys=(total..., "$(name)->Dst.D_g_ls_1"),
            lineshape=l2_lineshape)
    end
    push_chain!(rows, "Psi(4040)", "default", :DxD;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1)
    push_chain!(rows, "NR(0-)SPp", "default", :DxD;
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0",),
        lineshape=:nr_exp)
    push_chain!(rows, "NR(1.)PSp", "default", :DxD;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        coupling_keys=("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant_neg)
    push_chain!(rows, "NR(0-)SPm", "default", :DxD;
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant)
    push_chain!(rows, "NR(1-)PPm", "default", :DxD;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant)
    push_chain!(rows, "X0(2900)", "default", :dk;
        propagator_two_j=0, root_two_ls=(2, 2), daughter_two_ls=(0, 0),
        coupling_keys=("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0",),
        lineshape=:x2900_bwr_l0)
    total_x1 = ("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0",)
    daughter_x1 = (2, 0)
    push_chain!(rows, "X1(2900)", "l0", :dk;
        propagator_two_j=2, root_two_ls=(0, 0), daughter_two_ls=daughter_x1,
        coupling_keys=total_x1, lineshape=:x2900_bwr_l1, root_remove_particle2_phase=true)
    push_chain!(rows, "X1(2900)", "l1", :dk;
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_1"),
        lineshape=:x2900_bwr_l1, root_remove_particle2_phase=true)
    push_chain!(rows, "X1(2900)", "l2", :dk;
        propagator_two_j=2, root_two_ls=(4, 4), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_2"),
        lineshape=:x2900_bwr_l1, root_remove_particle2_phase=true)
    return DataFrame(rows)
end

function lineshape_param_keys(resonance_name::String, lineshape)
    keys = String[]
    base = lineshape_base(lineshape)
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

function parametrization_keys(resonance_name::String, coupling_keys, lineshape)
    return unique(vcat(collect(coupling_keys), lineshape_param_keys(resonance_name, lineshape)))
end

const resonance_chains_df_raw = build_resonance_chains_df()

# =============================================================================
# Block 2 — CascadeDecays construction
# (lineshape builders need event context; combined with block 3 in the event loop)
# =============================================================================

const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const dxd_topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))
const kinematic_task = KinematicTask((dxd_topology, dk_topology))

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

function decay_reference_mass(resonance_name::String, lineshape)
    base = lineshape_base(lineshape)
    below_threshold = base in (:bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
    m0 = nominal_mass[resonance_name]
    below_threshold || return m0
    return ad_hoc_mass(m0, nominal_mass["Dst"] + nominal_mass["D"], nominal_mass["Bp"] - nominal_mass["K"])
end

function chain_lineshape_static_matching_factor(resonance_name::String, lineshape)
    sign = lineshape_matching_sign(lineshape)
    base = lineshape_base(lineshape)
    if base in (:bwr_ls_l0, :bwr_ls_l0_below_threshold)
        return sign * bwr_ls_coupling_params(resonance_name).gamma0
    elseif base in (:bwr_ls_l2, :bwr_ls_l2_below_threshold)
        return sign * bwr_ls_coupling_params(resonance_name).gamma2
    elseif base in (:bwr_l1, :bwr_l2, :constant)
        return sign
    end
    return 1.0
end

function chain_static_matching_factor(row)
    return chain_vertex_matching_factor(row) *
           chain_lineshape_static_matching_factor(row.resonance_name, row.lineshape)
end

function enrich_resonance_chains_df!(df)
    df.coupling_value = resolve_coupling_keys.(df.coupling_keys)
    df.static_matching_factor = chain_static_matching_factor.(eachrow(df))
    bare = df.coupling_value .* lineshape_matching_sign.(df.lineshape)
    df.bare_coupling_re = real.(bare)
    df.bare_coupling_im = imag.(bare)
    df.coupling_param_keys = join.(df.coupling_keys, Ref(";"))
    df.bwr_l = bwr_decay_l.(df.lineshape)
    df.parametrization = [
        join(parametrization_keys(row.resonance_name, row.coupling_keys, row.lineshape), ";")
        for row in eachrow(df)
    ]
    return df
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

function build_chain_lineshape(ctx, row)
    resonance_name = row.resonance_name
    base = lineshape_base(row.lineshape)
    if base in (:bwr_ls_l0, :bwr_ls_l2, :bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
        below_threshold = base in (:bwr_ls_l0_below_threshold, :bwr_ls_l2_below_threshold)
        return build_bwr_ls_lineshape(ctx, resonance_name; below_threshold)
    elseif base in (:bwr_l1, :bwr_l2)
        return bwr_lineshape(
            ctx, nominal_mass[resonance_name], param_real(resonance_name * "_width"),
            bwr_decay_l(row.lineshape),
        )
    elseif base == :constant
        return ConstantLineshape(1.0 + 0.0im)
    elseif base == :nr_exp
        alpha = param_real("NR(0-)SPp_alpha")
        beta = param_real("NR(0-)SPp_beta")
        nr_factor = -exp(-(alpha + 1im * beta) * (mass(ctx.P_R)^2 - nominal_mass["NR(0-)SPp"]^2))
        return ConstantLineshape(nr_factor)
    elseif base in (:x2900_bwr_l0, :x2900_bwr_l1)
        return x2900_bwr_lineshape(ctx, resonance_name, bwr_decay_l(row.lineshape))
    end
    error("Unknown lineshape $(row.lineshape) for $(resonance_name).")
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

function build_dxd_chain(lineshape, two_j, root_two_ls, decay_two_ls; root_l=nothing, decay_l=nothing)
    return DecayChain(
        dxd_topology;
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

function dxd_vertex_matching_factor(resonance_name; root_l=nothing, decay_l=nothing, decay_m0=nothing)
    m_r = nominal_mass[resonance_name]
    decay_m0 = something(decay_m0, m_r)
    root = root_l === nothing ? 1.0 :
           nominal_vertex_matching_factor(root_l, 3.0, nominal_mass["Bp"], m_r, nominal_mass["K"])
    decay = decay_l === nothing ? 1.0 :
            nominal_vertex_matching_factor(decay_l, 3.0, decay_m0, nominal_mass["Dst"], nominal_mass["D"])
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

function chain_vertex_matching_factor(row)
    topology = Symbol(row.topology)
    root_l = row.root_l
    daughter_l = row.daughter_l
    if topology == :DxD
        return dxd_vertex_matching_factor(
            row.resonance_name;
            root_l=root_l,
            decay_l=daughter_l,
            decay_m0=decay_reference_mass(row.resonance_name, row.lineshape),
        )
    elseif topology == :dk
        return dk_vertex_matching_factor(row.resonance_name; root_l=root_l, dk_l=daughter_l)
    end
    error("Unknown topology $(topology).")
end

const resonance_chains_df = enrich_resonance_chains_df!(copy(resonance_chains_df_raw))

function resonance_chain_rows(name::String)
    resonance_chains_df[resonance_chains_df.resonance_name.==name, :]
end

function build_chain_from_row(ctx, row)
    lineshape = build_chain_lineshape(ctx, row)
    root_two_ls = (row.root_two_l, row.root_two_s)
    daughter_two_ls = (row.daughter_two_l, row.daughter_two_s)
    if Symbol(row.topology) == :dk
        return build_dk_chain(
            lineshape,
            row.propagator_two_j,
            root_two_ls,
            daughter_two_ls;
            root_l=row.root_l,
            dk_l=row.daughter_l,
            remove_root_particle2_phase=row.root_remove_particle2_phase,
        )
    end
    return build_dxd_chain(
        lineshape,
        row.propagator_two_j,
        root_two_ls,
        daughter_two_ls;
        root_l=row.root_l,
        decay_l=row.daughter_l,
    )
end

function build_resonance_cascade(resonance_name::String, ctx)
    rows = collect(eachrow(resonance_chain_rows(resonance_name)))
    chains = Tuple(build_chain_from_row(ctx, row) for row in rows)
    effective_couplings = Tuple(
        rows[i].coupling_value * rows[i].static_matching_factor / _propagator_spin_norm(chains[i])
        for i in eachindex(rows)
    )
    branch_names = Tuple(row.branch for row in rows)
    return CascadeDecay(
        chains,
        ctx.system,
        dxd_topology;
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
    cascade = build_resonance_cascade(name, ctx)
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
    println("Model input rows (one per chain branch): ", nrow(resonance_chains_df))
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
