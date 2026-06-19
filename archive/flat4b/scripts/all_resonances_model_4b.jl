# TF-PWA-aligned amplitude model for flat 4-body phase space.
# Standalone counterpart to all_resonances_model.jl: same block layout and chain
# table, but propagator channel masses and running breakup use event kinematics.

using CascadeDecays
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON
using Printf
import ThreeBodyDecays
using ThreeBodyDecays: Recoupling, RecouplingLS, VertexFunction, @jp_str

"""
    BuggyParticleTwoPhaseLS

Workaround recoupling that applies the Jacob–Wick particle-2 phase a second time so it
cancels the factor already built into CascadeDecays, matching TF-PWA (which omits it).
Not a physically correct recoupling on its own.
"""
struct BuggyParticleTwoPhaseLS <: Recoupling
    two_ls::Tuple{Int,Int}
end

function ThreeBodyDecays.amplitude(cs::BuggyParticleTwoPhaseLS, helicities, spins)
    _, _, two_j2 = spins
    _, two_lambda2 = helicities
    exponent_num = two_j2 - two_lambda2
    iseven(exponent_num) || error("particle-2 phase requires two_j2 - two_lambda2 to be even")
    phase = isodd(div(exponent_num, 2)) ? -1 : 1
    return phase * ThreeBodyDecays.amplitude(RecouplingLS(cs.two_ls), helicities, spins)
end

# =============================================================================
# Block 1 — model inputs
# =============================================================================

const flat4b_root = normpath(joinpath(@__DIR__, ".."))
const repo_root = normpath(joinpath(flat4b_root, "..", ".."))
const params_path = joinpath(repo_root, "Analysis", "final_params_full.json")

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

const WELL_SIZE = 3.0

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

tfpwa_breakup(m0, m1, m2) =
    sqrt(complex(((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)))

# =============================================================================
# Block 2 — event-dependent lineshape definitions
# =============================================================================

function ad_hoc_mass(m0, m_min, m_max)
    k = (m_max - m_min) / 2
    return k * (1 + tanh((2m0 - (m_max + m_min)) / k / 4)) + m_min
end

dxd_adhoc_q0_mass(name::String) = ad_hoc_mass(
    nominal_mass[name],
    nominal_mass["Dst"] + nominal_mass["D"],
    nominal_mass["Bp"] - nominal_mass["K"],
)

const BW_DECAY_L = Dict(
    :bwr_l1 => 1,
    :bwr_l2 => 2,
    :x2900_bwr_l0 => 0,
    :x2900_bwr_l1 => 1,
)

const MC_BW_GAMMA = Dict(
    :bwr_ls_l0 => :gamma0,
    :adhoc_q0_bwr_ls_l0 => :gamma0,
    :bwr_ls_l2 => :gamma2,
    :adhoc_q0_bwr_ls_l2 => :gamma2,
)

const ADHOC_Q0_BASES = Set([:adhoc_q0_bwr_ls_l0, :adhoc_q0_bwr_ls_l2])

function lineshape_spec(lineshape::Symbol)
    return (;
        base=lineshape,
        bwr_l=get(BW_DECAY_L, lineshape, nothing),
        mc_gamma=get(MC_BW_GAMMA, lineshape, nothing),
        adhoc_q0=lineshape in ADHOC_Q0_BASES,
    )
end

function bwr_ls_coupling_params(name::String)
    theta0 = param_real(name * "_theta0")
    return (; gamma0=cos(theta0), gamma2=sin(theta0))
end

function dxd_channel_masses(ctx)
    return mass(ctx.P_Dst), mass(ctx.pDminus)
end

function dk_channel_masses(ctx)
    return mass(ctx.pDminus), mass(ctx.pKplus)
end

function bwr_lineshape_4b(ctx, m0, width, l, sign)
    q0 = real(tfpwa_breakup(m0, nominal_mass["Dst"], nominal_mass["D"]))
    ff = BlattWeisskopf{l}(WELL_SIZE)
    gsq = m0 * width / (2q0) * m0 / ff(q0)^2
    ma, mb = dxd_channel_masses(ctx)
    return sign * MultichannelBreitWigner(m0, [(; gsq, ma, mb, l, d=WELL_SIZE)])
end

function bwr_ls_lineshapes_4b(ctx, name::String, sign; below_threshold=false)
    m0 = nominal_mass[name]
    q0_mass = below_threshold ? dxd_adhoc_q0_mass(name) : m0
    q0 = real(tfpwa_breakup(q0_mass, nominal_mass["Dst"], nominal_mass["D"]))
    gamma0 = cos(param_real(name * "_theta0"))
    gamma2 = sin(param_real(name * "_theta0"))
    ff0 = BlattWeisskopf{0}(WELL_SIZE)
    ff2 = BlattWeisskopf{2}(WELL_SIZE)
    ma, mb = dxd_channel_masses(ctx)
    channels = [
        (; gsq=param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma0^2 / ff0(q0)^2,
           ma, mb, l=0, d=WELL_SIZE),
        (; gsq=param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma2^2 / ff2(q0)^2,
           ma, mb, l=2, d=WELL_SIZE),
    ]
    bw = sign * MultichannelBreitWigner(m0, channels)
    breakup_from_sigma = sigma -> tfpwa_breakup(sqrt(sigma), ma, mb)
    return bw * gamma0, bw * (ff2(breakup_from_sigma) * (gamma2 / ff2(q0)))
end

function x2900_bwr_lineshape_4b(ctx, name::String, l)
    q0 = real(tfpwa_breakup(nominal_mass[name], nominal_mass["D"], nominal_mass["K"]))
    ff = BlattWeisskopf{l}(WELL_SIZE)
    gsq = nominal_mass[name] * param_real(name * "_width") / (2q0) * nominal_mass[name] / ff(q0)^2
    ma, mb = dk_channel_masses(ctx)
    return MultichannelBreitWigner(nominal_mass[name], [(; gsq, ma, mb, l, d=WELL_SIZE)])
end

function nr_exp_factor_4b(ctx)
    alpha = param_real("NR(0-)SPp_alpha")
    beta = param_real("NR(0-)SPp_beta")
    return -exp(-(alpha + 1im * beta) * (mass(ctx.P_R)^2 - nominal_mass["NR(0-)SPp"]^2))
end

function build_chain_lineshape_4b(ctx, row)
    name = row.resonance_name
    spec = lineshape_spec(row.lineshape)
    sign = magic_sign(name) + 0im

    if spec.base in (:bwr_ls_l0, :bwr_ls_l2)
        l0, = bwr_ls_lineshapes_4b(ctx, name, sign)
        return l0
    elseif spec.base in (:adhoc_q0_bwr_ls_l0, :adhoc_q0_bwr_ls_l2)
        l0, = bwr_ls_lineshapes_4b(ctx, name, sign; below_threshold=true)
        return l0
    elseif spec.base == :bwr_l1
        return bwr_lineshape_4b(ctx, nominal_mass[name], param_real(name * "_width"), 1, sign)
    elseif spec.base == :bwr_l2
        return bwr_lineshape_4b(ctx, nominal_mass[name], param_real(name * "_width"), 2, sign)
    elseif spec.base == :constant
        return ConstantLineshape(sign)
    elseif spec.base == :nr_exp
        return ConstantLineshape(nr_exp_factor_4b(ctx))
    elseif spec.base in (:x2900_bwr_l0, :x2900_bwr_l1)
        return x2900_bwr_lineshape_4b(ctx, name, spec.bwr_l)
    end
    error("Unknown lineshape $(row.lineshape) for $(name).")
end

# =============================================================================
# Block 3 — resonance chain table
# =============================================================================

function production_coupling_key(name::String)
    name == "X(3940)(1.)" &&
        return "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0"
    return "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
end

function build_resonance_chains_df()
    rows = NamedTuple[]
    total_x3872 = ("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0",)
    push!(rows, (
        resonance_name="X(3872)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        root_remove_particle2_phase=false,
        coupling_keys=total_x3872, lineshape=:adhoc_q0_bwr_ls_l0,
    ))
    push!(rows, (
        resonance_name="X(3872)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
        root_remove_particle2_phase=false,
        coupling_keys=(total_x3872..., "X(3872)->Dst.D_g_ls_1"),
        lineshape=:adhoc_q0_bwr_ls_l2,
    ))
    push!(rows, (
        resonance_name="X(3915)(0-)", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
    ))
    push!(rows, (
        resonance_name="chi(c2)(3930)", topology=:DxD,
        propagator_two_j=4, root_two_ls=(4, 4), daughter_two_ls=(4, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l2,
    ))
    for name in ("X(3940)(1.)", "X(3993)", "X(4300)")
        total = (production_coupling_key(name),)
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
            root_remove_particle2_phase=false,
            coupling_keys=total, lineshape=:bwr_ls_l0,
        ))
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
            root_remove_particle2_phase=false,
            coupling_keys=(total..., "$(name)->Dst.D_g_ls_1"),
            lineshape=:bwr_ls_l2,
        ))
    end
    push!(rows, (
        resonance_name="Psi(4040)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPp", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0",),
        lineshape=:nr_exp,
    ))
    push!(rows, (
        resonance_name="NR(1.)PSp", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPm", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="NR(1-)PPm", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="X0(2900)", topology=:dk,
        propagator_two_j=0, root_two_ls=(2, 2), daughter_two_ls=(0, 0),
        root_remove_particle2_phase=false,
        coupling_keys=("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0",),
        lineshape=:x2900_bwr_l0,
    ))
    total_x1 = ("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0",)
    daughter_x1 = (2, 0)
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(0, 0), daughter_two_ls=daughter_x1,
        root_remove_particle2_phase=true,
        coupling_keys=total_x1, lineshape=:x2900_bwr_l1,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=daughter_x1,
        root_remove_particle2_phase=true,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_1"),
        lineshape=:x2900_bwr_l1,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(4, 4), daughter_two_ls=daughter_x1,
        root_remove_particle2_phase=true,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_2"),
        lineshape=:x2900_bwr_l1,
    ))
    return DataFrame(rows)
end

const resonance_chains_df = build_resonance_chains_df()

function resonance_chain_rows(name::String)
    resonance_chains_df[resonance_chains_df.resonance_name.==name, :]
end

# =============================================================================
# Block 4 — nominal vertex matching (TF-PWA mismatch factors)
# =============================================================================

const MAGIC_SIGNS = Dict{String,Float64}(
    "X(3872)" => -1.0,
    "X(3915)(0-)" => -1.0,
    "chi(c2)(3930)" => -1.0,
    "X(3940)(1.)" => 1.0,
    "X(3993)" => -1.0,
    "Psi(4040)" => 1.0,
    "X(4300)" => 1.0,
    "NR(0-)SPp" => 1.0,
    "NR(1.)PSp" => -1.0,
    "NR(0-)SPm" => 1.0,
    "NR(1-)PPm" => 1.0,
    "X0(2900)" => 1.0,
    "X1(2900)" => 1.0,
)
@assert Set(keys(MAGIC_SIGNS)) == Set(all_resonance_names)

magic_sign(resonance_name::String) = MAGIC_SIGNS[resonance_name]

const DSTAR_PROPAGATOR_TWO_J = 2
const ROOT_MDEP_RESONANCES = Set(["X(3872)", "X(3940)(1.)", "X(3993)", "X(4300)"])

vertex_l(two_ls) = div(two_ls[1], 2)

nominal_vertex_matching_factor(l, d, m0, m1, m2) = 1 / BlattWeisskopf{l}(d)(m0^2, m1^2, m2^2)

function dxd_vertex_matching_factor(resonance_name; root_l, decay_l, decay_m0=nothing)
    m_r = nominal_mass[resonance_name]
    decay_m0 = something(decay_m0, m_r)
    root = nominal_vertex_matching_factor(root_l, WELL_SIZE, nominal_mass["Bp"], m_r, nominal_mass["K"])
    decay = nominal_vertex_matching_factor(decay_l, WELL_SIZE, decay_m0, nominal_mass["Dst"], nominal_mass["D"])
    return root * decay
end

function dk_vertex_matching_factor(resonance_name; root_l, dk_l)
    m_r = nominal_mass[resonance_name]
    root = nominal_vertex_matching_factor(root_l, WELL_SIZE, nominal_mass["Bp"], m_r, nominal_mass["Dst"])
    dk = nominal_vertex_matching_factor(dk_l, WELL_SIZE, m_r, nominal_mass["D"], nominal_mass["K"])
    return root * dk
end

chain_propagator_spin_norm(row) =
    sqrt(DSTAR_PROPAGATOR_TWO_J + 1) * sqrt(row.propagator_two_j + 1)

function vertex_blatt_ls(row)
    name = row.resonance_name
    spec = lineshape_spec(row.lineshape)
    if spec.base in (:bwr_ls_l0, :bwr_ls_l2, :adhoc_q0_bwr_ls_l0, :adhoc_q0_bwr_ls_l2)
        return nothing, nothing
    elseif name == "NR(0-)SPp"
        return nothing, 1
    elseif name == "NR(1.)PSp"
        return 1, nothing
    elseif name == "NR(0-)SPm"
        return nothing, 1
    end
    return vertex_l(row.root_two_ls), vertex_l(row.daughter_two_ls)
end

function row_correction_factor(row)
    root_l, daughter_l = vertex_blatt_ls(row)
    spin = chain_propagator_spin_norm(row)
    name = row.resonance_name
    spec = lineshape_spec(row.lineshape)

    if row.topology == :DxD
        if spec.base in (:bwr_l1, :bwr_l2)
            return dxd_vertex_matching_factor(name; root_l=root_l, decay_l=daughter_l) / spin
        elseif spec.base == :nr_exp
            return nominal_vertex_matching_factor(1, WELL_SIZE, nominal_mass["NR(0-)SPp"], nominal_mass["Dst"], nominal_mass["D"]) / spin
        elseif spec.base == :constant && daughter_l == 1 && root_l === nothing
            return nominal_vertex_matching_factor(1, WELL_SIZE, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"]) / spin
        elseif spec.base == :constant && root_l == 1 && daughter_l === nothing
            return nominal_vertex_matching_factor(1, WELL_SIZE, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) / spin
        elseif spec.base == :constant && root_l == 1 && daughter_l == 1
            return (
                nominal_vertex_matching_factor(1, WELL_SIZE, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) *
                nominal_vertex_matching_factor(1, WELL_SIZE, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"]) / spin
            )
        end
    elseif row.topology == :dk && spec.base in (:x2900_bwr_l0, :x2900_bwr_l1)
        return dk_vertex_matching_factor(name; root_l=root_l, dk_l=daughter_l) / spin
    end
    error("No correction rule for $(name) / $(row.lineshape).")
end

function root_mdep_factor(ctx, resonance_name::String)
    root_ff = BlattWeisskopf{1}(WELL_SIZE)
    root_ff(tfpwa_breakup(mass(ctx.P_B), mass(ctx.P_R), mass(ctx.pKplus))) /
    root_ff(tfpwa_breakup(nominal_mass["Bp"], nominal_mass[resonance_name], nominal_mass["K"]))
end

# =============================================================================
# Block 5 — event context and amplitude evaluation
# =============================================================================

const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const dxd_topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))

struct EventContext4b
    sampled_p4::Dict{String, Vector{Float64}}
    pDminus::FourVector
    pD0::FourVector
    pKplus::FourVector
    piplus::FourVector
    P_Dst::FourVector
    P_R::FourVector
    P_B::FourVector
    system::CascadeSystem
    x_dxd::CascadeKinematics
end

function _fourvector_from_sample(name::String, sampled_p4)
    v = sampled_p4[name]
    return FourVector(v[2], v[3], v[4]; E=v[1])
end

function event_context_from_p4(sampled_p4::Dict{String, Vector{Float64}})
    pDminus = _fourvector_from_sample("D", sampled_p4)
    pD0 = _fourvector_from_sample("D0", sampled_p4)
    pKplus = _fourvector_from_sample("K", sampled_p4)
    piplus = _fourvector_from_sample("pi", sampled_p4)
    P_Dst = pD0 + piplus
    P_R = P_Dst + pDminus
    P_B = P_R + pKplus
    objs = (pD0, piplus, pDminus, pKplus)
    system = CascadeSystem(external_spins, SystemMasses(mass.(objs)...; m0=mass(P_B)))
    return EventContext4b(
        sampled_p4,
        pDminus,
        pD0,
        pKplus,
        piplus,
        P_Dst,
        P_R,
        P_B,
        system,
        CascadeKinematics(dxd_topology, objs),
    )
end

function event_context_from_row(row)
    sampled_p4 = Dict(
        "D0" => [row.D0_E, row.D0_px, row.D0_py, row.D0_pz],
        "pi" => [row.pip_E, row.pip_px, row.pip_py, row.pip_pz],
        "D" => [row.Dm_E, row.Dm_px, row.Dm_py, row.Dm_pz],
        "K" => [row.Kp_E, row.Kp_px, row.Kp_py, row.Kp_pz],
    )
    return event_context_from_p4(sampled_p4)
end

function dk_event_context(ctx::EventContext4b)
    objs = (ctx.pD0, ctx.piplus, ctx.pDminus, ctx.pKplus)
    P_DK = ctx.pDminus + ctx.pKplus
    return (
        pDminus=ctx.pDminus,
        pD0=ctx.pD0,
        pKplus=ctx.pKplus,
        piplus=ctx.piplus,
        P_Dst=ctx.P_Dst,
        P_DK=P_DK,
        P_B=ctx.P_Dst + P_DK,
        system=ctx.system,
        x=CascadeKinematics(dk_topology, objs),
    )
end

function chain_amplitude_4b(ctx::EventContext4b, lineshape, two_j, root_two_ls, decay_two_ls; root_l=nothing, decay_l=nothing)
    root_vertex = root_l === nothing ?
        VertexFunction(RecouplingLS(root_two_ls)) :
        VertexFunction(RecouplingLS(root_two_ls), BlattWeisskopf{root_l}(WELL_SIZE))
    decay_vertex = decay_l === nothing ?
        VertexFunction(RecouplingLS(decay_two_ls)) :
        VertexFunction(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(WELL_SIZE))
    chain = DecayChain(
        dxd_topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => Propagator(two_j, lineshape),
        ),
        vertices=(
            (((1, 2), 3), 4) => root_vertex,
            ((1, 2), 3) => decay_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, ctx.system, ctx.x_dxd, external_spins)
end

function dk_chain_amplitude_4b(dk_ctx, lineshape, two_j, root_two_ls, dk_two_ls; root_l=nothing, dk_l=nothing, remove_root_particle2_phase=false)
    root_recoupling = remove_root_particle2_phase ? BuggyParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    root_vertex = root_l === nothing ?
        VertexFunction(root_recoupling) :
        VertexFunction(root_recoupling, BlattWeisskopf{root_l}(WELL_SIZE))
    dk_vertex = dk_l === nothing ?
        VertexFunction(RecouplingLS(dk_two_ls)) :
        VertexFunction(RecouplingLS(dk_two_ls), BlattWeisskopf{dk_l}(WELL_SIZE))
    chain = DecayChain(
        dk_topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => Propagator(two_j, lineshape),
        ),
        vertices=(
            ((1, 2), (3, 4)) => root_vertex,
            (3, 4) => dk_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, dk_ctx.system, dk_ctx.x, external_spins)
end

function evaluate_chain_row(ctx::EventContext4b, row, lineshape)
    root_l, daughter_l = vertex_blatt_ls(row)
    if row.topology == :dk
        dk_ctx = dk_event_context(ctx)
        return dk_chain_amplitude_4b(
            dk_ctx,
            lineshape,
            row.propagator_two_j,
            row.root_two_ls,
            row.daughter_two_ls;
            root_l=root_l,
            dk_l=daughter_l,
            remove_root_particle2_phase=row.root_remove_particle2_phase,
        )
    end
    return chain_amplitude_4b(
        ctx,
        lineshape,
        row.propagator_two_j,
        row.root_two_ls,
        row.daughter_two_ls;
        root_l=root_l,
        decay_l=daughter_l,
    )
end

function single_row_amplitude(ctx::EventContext4b, row)
    lineshape = build_chain_lineshape_4b(ctx, row)
    raw = evaluate_chain_row(ctx, row, lineshape)
    return resolve_coupling_keys(row.coupling_keys) * row_correction_factor(row) * raw
end

function root_mdep_resonance_amplitude(ctx::EventContext4b, name::String)
    rows = collect(eachrow(resonance_chain_rows(name)))
    sign = magic_sign(name) + 0im
    below_threshold = name == "X(3872)"
    l0, l2 = bwr_ls_lineshapes_4b(ctx, name, sign; below_threshold)
    rmdep = root_mdep_factor(ctx, name)
    amp_l0 = rmdep * chain_amplitude_4b(ctx, l0, rows[1].propagator_two_j, rows[1].root_two_ls, rows[1].daughter_two_ls)
    amp_l2 = rmdep * chain_amplitude_4b(ctx, l2, rows[2].propagator_two_j, rows[2].root_two_ls, rows[2].daughter_two_ls)
    total = resolve_coupling_keys((rows[1].coupling_keys[1],))
    ls1 = resolve_coupling_keys((rows[2].coupling_keys[2],))
    spin = chain_propagator_spin_norm(rows[1])
    return total * (amp_l0 + ls1 * amp_l2) / spin
end

function x1_2900_amplitude(ctx::EventContext4b)
    rows = collect(eachrow(resonance_chain_rows("X1(2900)")))
    dk_ctx = dk_event_context(ctx)
    lineshape = x2900_bwr_lineshape_4b(dk_ctx, "X1(2900)", 1)
    raw_l0 = dk_chain_amplitude_4b(dk_ctx, lineshape, 2, (0, 0), (2, 0); root_l=0, dk_l=1, remove_root_particle2_phase=true)
    raw_l1 = dk_chain_amplitude_4b(dk_ctx, lineshape, 2, (2, 2), (2, 0); root_l=1, dk_l=1, remove_root_particle2_phase=true)
    raw_l2 = dk_chain_amplitude_4b(dk_ctx, lineshape, 2, (4, 4), (2, 0); root_l=2, dk_l=1, remove_root_particle2_phase=true)
    spin = chain_propagator_spin_norm(rows[1])
    c0 = dk_vertex_matching_factor("X1(2900)"; root_l=0, dk_l=1) / spin
    c1 = dk_vertex_matching_factor("X1(2900)"; root_l=1, dk_l=1) / spin
    c2 = dk_vertex_matching_factor("X1(2900)"; root_l=2, dk_l=1) / spin
    coherent = raw_l0 * c0 +
               param_complex("Bp->X1(2900).Dst_g_ls_1") * raw_l1 * c1 +
               param_complex("Bp->X1(2900).Dst_g_ls_2") * raw_l2 * c2
    return param_complex("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0") * coherent
end

function resonance_amplitude(ctx::EventContext4b, name::String)
    name in ROOT_MDEP_RESONANCES && return root_mdep_resonance_amplitude(ctx, name)
    name == "X1(2900)" && return x1_2900_amplitude(ctx)
    return sum(single_row_amplitude(ctx, row) for row in eachrow(resonance_chain_rows(name)))
end

function all_resonances_amplitude(ctx::EventContext4b, names=all_resonance_names)
    sum(resonance_amplitude(ctx, name) for name in names)
end

# Backward-compatible aliases for the comparison notebook.
const event_context = event_context_from_p4
const selected_cd_amplitude = resonance_amplitude
const all_resonance_cd_amplitude = all_resonances_amplitude
