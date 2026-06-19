using CascadeDecays
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON
using Printf
using StaticArrays
import ThreeBodyDecays
using ThreeBodyDecays: Recoupling, RecouplingLS, @jp_str

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
# Block 1 — model inputs (no CascadeDecays types or API)
# =============================================================================

const repo_root = normpath(joinpath(@__DIR__, ".."))
const data_dir = joinpath(repo_root, "data")
const params_path = joinpath(data_dir, "final_params_full.json")

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

# =============================================================================
# Block 2 — lineshape definitions
# Block 2a — custom lineshapes
# =============================================================================

"""
    TFPWAMultichannelBreitWigner

TFPWA-aligned multichannel Breit–Wigner: width term carries an extra factor of `σ`
relative to [`MultichannelBreitWigner`](@ref), i.e. `gsq * σ * 2p/√σ * F_ℓ²` instead of
`gsq * 2p/√σ * F_ℓ²`.  Store `gsq` calibrated without that `σ` factor.
"""
struct TFPWAMultichannelBreitWigner{N} <: HadronicLineshapes.AbstractFlexFunc
    m::Float64
    channels::SVector{N,<:NamedTuple{(:gsq, :ma, :mb, :l, :d)}}
end

function TFPWAMultichannelBreitWigner(
    m::Real,
    channels::Vector{<:NamedTuple{(:gsq, :ma, :mb, :l, :d)}},
)
    N = length(channels)
    return TFPWAMultichannelBreitWigner(m, SVector{N}(channels...))
end

function (bw::TFPWAMultichannelBreitWigner)(σ::Number)
    m0 = bw.m
    mΓ = sum(bw.channels) do channel
        gsq, ma, mb, l, d = channel.gsq, channel.ma, channel.mb, channel.l, channel.d
        FF = BlattWeisskopf{l}(d)
        p = breakup(sqrt(σ), ma, mb)
        gsq * σ * 2p / sqrt(σ) * FF(p)^2
    end
    HadronicLineshapes.BW(σ, m0, mΓ / m0)
end
(bw::TFPWAMultichannelBreitWigner)(σ::Real) = bw(σ + 1im * eps())

"""
    NRExpLineshape

Nonresonant exponential: `-exp(-αβ * (σ - m0²))` with invariant mass squared `σ`.
"""
struct NRExpLineshape <: HadronicLineshapes.AbstractFlexFunc
    αβ::ComplexF64
    m0::Float64
end

function (ls::NRExpLineshape)(σ::Number)
    -exp(-ls.αβ * (σ - ls.m0^2))
end
(ls::NRExpLineshape)(σ::Real) = ls(σ + 1im * eps())


function nr_exp_lineshape()
    alpha = param_real("NR(0-)SPp_alpha")
    beta = param_real("NR(0-)SPp_beta")
    return NRExpLineshape(alpha + 1im * beta, nominal_mass["NR(0-)SPp"])
end

# Block 2b — lineshape helpers
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

bwr_ls_q0(name::String) = breakup(nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
bwr_ls_adhoc_q0(name::String) =
    breakup(dxd_adhoc_q0_mass(name), nominal_mass["Dst"], nominal_mass["D"])

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

dxd_bwr_lineshape(name::String, l) =
    BreitWigner(nominal_mass[name], param_real(name * "_width"), nominal_mass["Dst"], nominal_mass["D"], l, WELL_SIZE)

dk_bwr_lineshape(name::String, l) =
    BreitWigner(nominal_mass[name], param_real(name * "_width"), nominal_mass["D"], nominal_mass["K"], l, WELL_SIZE)

function bwr_ls_coupling_params(name::String)
    theta0 = param_real(name * "_theta0")
    return (; gamma0=cos(theta0), gamma2=sin(theta0))
end

function dxd_tfpwa_multichannel_bwr_lineshape(name::String, q0::Real)
    (; gamma0, gamma2) = bwr_ls_coupling_params(name)
    ff0 = BlattWeisskopf{0}(WELL_SIZE)
    ff2 = BlattWeisskopf{2}(WELL_SIZE)
    Γ0 = param_real(name * "_width")
    gsq_0 = Γ0 / (2q0) * gamma0^2 / ff0(q0)^2
    gsq_2 = Γ0 / (2q0) * gamma2^2 / ff2(q0)^2
    ma = nominal_mass["Dst"]
    mb = nominal_mass["D"]
    channels = [
        (; gsq=gsq_0, ma, mb, l=0, d=WELL_SIZE),
        (; gsq=gsq_2, ma, mb, l=2, d=WELL_SIZE),
    ]
    return TFPWAMultichannelBreitWigner(nominal_mass[name], channels)
end

function decay_reference_mass(resonance_name::String, lineshape)
    lineshape_spec(lineshape).adhoc_q0 && return dxd_adhoc_q0_mass(resonance_name)
    return nominal_mass[resonance_name]
end

function build_chain_lineshape(row)
    resonance_name = row.resonance_name
    spec = lineshape_spec(row.lineshape)
    if spec.base in (:bwr_ls_l0, :bwr_ls_l2)
        return dxd_tfpwa_multichannel_bwr_lineshape(resonance_name, bwr_ls_q0(resonance_name))
    elseif spec.adhoc_q0
        return dxd_tfpwa_multichannel_bwr_lineshape(resonance_name, bwr_ls_adhoc_q0(resonance_name))
    elseif spec.base in (:bwr_l1, :bwr_l2)
        return dxd_bwr_lineshape(resonance_name, spec.bwr_l)
    elseif spec.base == :constant
        return ConstantLineshape(1.0 + 0.0im)
    elseif spec.base == :nr_exp
        return nr_exp_lineshape()
    elseif spec.base in (:x2900_bwr_l0, :x2900_bwr_l1)
        return dk_bwr_lineshape(resonance_name, spec.bwr_l)
    end
    error("Unknown lineshape $(row.lineshape) for $(resonance_name).")
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

const resonance_chains_df_raw = build_resonance_chains_df()

# =============================================================================
# Block 4 — matching factors (per chain)
#
# Matching is evaluated per chain (one row of `resonance_chains_df`).  Total
# matching is the product of vertex and lineshape contributions:
#
#     M_chain = M_sign × M_vertex × M_lineshape / N_propagator_spin
#
# Dependencies:
#   M_vertex           — topology, resonance_name, root_two_ls, daughter_two_ls, lineshape
#                        (lineshape sets decay reference mass for adhoc-q0 DxD chains)
#   M_lineshape        — resonance_name, lineshape (multichannel γ₀/γ₂ split)
#   M_sign             — `MAGIC_SIGNS[resonance_name]` (TF-PWA overall sign)
#   N_propagator_spin  — propagator_two_j (CascadeDecays v0.1.0 spin norm; D* line fixed at jp"1+")
# =============================================================================

# Overall ±1 amplitude sign per resonance for TF-PWA alignment (see
# archive/flat4b/notebooks/all_resonances_sampled_comparison.jl).  Not the same as Resonances.yml `C`.
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

const DSTAR_PROPAGATOR_TWO_J = 2  # jp"1+" on D* (Dst) line (1, 2) in every chain

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

function chain_vertex_matching_factor(row)
    root_l = vertex_l(row.root_two_ls)
    daughter_l = vertex_l(row.daughter_two_ls)
    if row.topology == :DxD
        return dxd_vertex_matching_factor(
            row.resonance_name;
            root_l=root_l,
            decay_l=daughter_l,
            decay_m0=decay_reference_mass(row.resonance_name, row.lineshape),
        )
    elseif row.topology == :dk
        return dk_vertex_matching_factor(row.resonance_name; root_l=root_l, dk_l=daughter_l)
    end
    error("Unknown topology $(row.topology).")
end

function chain_lineshape_matching_factor(resonance_name::String, lineshape)
    spec = lineshape_spec(lineshape)
    if spec.mc_gamma === :gamma0
        return bwr_ls_coupling_params(resonance_name).gamma0
    elseif spec.mc_gamma === :gamma2
        return bwr_ls_coupling_params(resonance_name).gamma2
    end
    return 1.0
end

chain_propagator_spin_norm(row) =
    sqrt(DSTAR_PROPAGATOR_TWO_J + 1) * sqrt(row.propagator_two_j + 1)

function chain_matching_factor(row)
    return (
        magic_sign(row.resonance_name) *
        chain_vertex_matching_factor(row) *
        chain_lineshape_matching_factor(row.resonance_name, row.lineshape) /
        chain_propagator_spin_norm(row)
    )
end

# =============================================================================
# Block 4b — informational summaries (not used in computation)
# =============================================================================

function info_lineshape_param_keys(resonance_name::String, lineshape)
    spec = lineshape_spec(lineshape)
    keys = String[]
    if spec.base in (:bwr_l1, :bwr_l2) || spec.mc_gamma !== nothing
        push!(keys, resonance_name * "_width")
        spec.mc_gamma !== nothing && push!(keys, resonance_name * "_theta0")
    elseif spec.base == :nr_exp
        append!(keys, ["NR(0-)SPp_alpha", "NR(0-)SPp_beta"])
    elseif spec.base in (:x2900_bwr_l0, :x2900_bwr_l1)
        push!(keys, resonance_name * "_width")
    end
    return keys
end

function info_parametrization_keys(resonance_name::String, coupling_keys, lineshape)
    return unique(vcat(collect(coupling_keys), info_lineshape_param_keys(resonance_name, lineshape)))
end

function info_enrich_resonance_chains_df!(df)
    specs = lineshape_spec.(df.lineshape)
    df.info_nominal_mass = [nominal_mass[name] for name in df.resonance_name]
    df.info_bare_coupling_re = real.(df.coupling_value)
    df.info_bare_coupling_im = imag.(df.coupling_value)
    df.info_coupling_param_keys = join.(df.coupling_keys, Ref(";"))
    df.info_bwr_l = [spec.bwr_l for spec in specs]
    df.info_parametrization = [
        join(info_parametrization_keys(row.resonance_name, row.coupling_keys, row.lineshape), ";")
        for row in eachrow(df)
    ]
    return df
end

# =============================================================================
# Block 5 — CascadeDecays construction
# =============================================================================

const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const dxd_topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))
const kinematic_task = KinematicTask((dxd_topology, dk_topology))
const standard_system = CascadeSystem(
    external_spins,
    SystemMasses(
        nominal_mass["D0"],
        nominal_mass["pi"],
        nominal_mass["D"],
        nominal_mass["K"];
        m0=nominal_mass["Bp"],
    ),
)

function event_point(row)
    pDminus = FourVector(row.Dm_px, row.Dm_py, row.Dm_pz; E=row.Dm_E)
    pD0 = FourVector(row.D0_px, row.D0_py, row.D0_pz; E=row.D0_E)
    pKplus = FourVector(row.Kp_px, row.Kp_py, row.Kp_pz; E=row.Kp_E)
    piplus = FourVector(row.pip_px, row.pip_py, row.pip_pz; E=row.pip_E)
    return KinematicPoint(kinematic_task, (pD0, piplus, pDminus, pKplus))
end

function enrich_resonance_chains_df!(df)
    df.coupling_value = resolve_coupling_keys.(df.coupling_keys)
    df.matching_factor = chain_matching_factor.(eachrow(df))
    info_enrich_resonance_chains_df!(df)
    return df
end

function build_dxd_chain(lineshape, two_j, root_two_ls, decay_two_ls, root_l, decay_l)
    return DecayChain(
        dxd_topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => Propagator(two_j, lineshape),
        ),
        vertices=(
            (((1, 2), 3), 4) => Vertex(RecouplingLS(root_two_ls), BlattWeisskopf{root_l}(WELL_SIZE)),
            ((1, 2), 3) => Vertex(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function build_dk_chain(
    lineshape,
    two_j,
    root_two_ls,
    dk_two_ls,
    root_l,
    dk_l;
    remove_root_particle2_phase=false,
)
    root_recoupling = remove_root_particle2_phase ? BuggyParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    return DecayChain(
        dk_topology;
        propagators=(
            (1, 2) => Propagator(jp"1+", ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => Propagator(two_j, lineshape),
        ),
        vertices=(
            ((1, 2), (3, 4)) => Vertex(root_recoupling, BlattWeisskopf{root_l}(WELL_SIZE)),
            (3, 4) => Vertex(RecouplingLS(dk_two_ls), BlattWeisskopf{dk_l}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

const resonance_chains_df = enrich_resonance_chains_df!(copy(resonance_chains_df_raw))

function resonance_chain_rows(name::String)
    resonance_chains_df[resonance_chains_df.resonance_name.==name, :]
end

function build_chain_from_row(row)
    lineshape = build_chain_lineshape(row)
    root_l = vertex_l(row.root_two_ls)
    daughter_l = vertex_l(row.daughter_two_ls)
    if row.topology == :dk
        return build_dk_chain(
            lineshape,
            row.propagator_two_j,
            row.root_two_ls,
            row.daughter_two_ls,
            root_l,
            daughter_l;
            remove_root_particle2_phase=row.root_remove_particle2_phase,
        )
    end
    return build_dxd_chain(
        lineshape,
        row.propagator_two_j,
        row.root_two_ls,
        row.daughter_two_ls,
        root_l,
        daughter_l,
    )
end

function chain_name(resonance_name::String, row)
    "$(resonance_name)_L$(vertex_l(row.root_two_ls))_d$(vertex_l(row.daughter_two_ls))"
end

function resonance_chain_names(resonance_name::String)
    [chain_name(resonance_name, row) for row in eachrow(resonance_chain_rows(resonance_name))]
end

function build_resonance_cascade(resonance_name::String)
    rows = collect(eachrow(resonance_chain_rows(resonance_name)))
    chains = Tuple(build_chain_from_row(row) for row in rows)
    effective_couplings = Tuple(
        rows[i].coupling_value * rows[i].matching_factor
        for i in eachindex(rows)
    )
    names = Tuple(chain_name(resonance_name, row) for row in rows)
    return CascadeDecay(
        chains,
        standard_system,
        dxd_topology;
        couplings=effective_couplings,
        names=names,
    )
end

function build_all_resonance_cascade(names=all_resonance_names)
    merge([build_resonance_cascade(name) for name in names]...)
end
