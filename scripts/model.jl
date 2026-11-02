# B+ -> D- D*+ K+ amplitude model of the LHCb TF-PWA analysis, built with CascadeDecays.jl.
# Scripts load it with `include("model.jl")` (tests: `include("../scripts/model.jl")`);
# couplings are read from data/final_params_full.json when the file is included.
using CascadeDecays
import CascadeDecays.ThreeBodyDecays
using CascadeDecays.ThreeBodyDecays: Recoupling, RecouplingLS
using CascadeDecaysIO: GenericFunction, NamedLineshape
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON

# ---------------------------------------------------------------------------
# Parameters: masses, Blatt-Weisskopf radius, TF-PWA fit parameters
# ---------------------------------------------------------------------------

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

# D*+ on its mass shell (TF-PWA: fixed D* mass): flat-top delta profile of the
# D* line, |g|^2 = exp(-((m(D0 pi) - m(D*)) / w)^n), see dstar_delta_lineshape
const DSTAR_DELTA_WIDTH = 1e-6   # GeV
const DSTAR_DELTA_ORDER = 16

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

# ---------------------------------------------------------------------------
# Lineshapes (TF-PWA conventions)
# ---------------------------------------------------------------------------

# Lineshapes without a standard type of amplitude-serialization are GenericFunctions of
# CascadeDecaysIO: the expression in `sigma` (invariant mass squared of the node) is what
# Julia evaluates and what CascadeDecaysIO writes to the JSON. Masses enter as written in
# the model ("4.35^2"), so every number can be traced to its parameter.

_expr_number(x) = repr(Float64(x))

# χ_l(z²) of the Blatt–Weisskopf factor F_l² = z^(2l) / χ_l(z²), as in HadronicLineshapes
const _BLATT_WEISSKOPF_CHI = Dict(
    1 => (1, 1),
    2 => (9, 3, 1),
    3 => (225, 45, 6, 1),
    4 => (11025, 1575, 135, 10, 1),
)

function _blatt_weisskopf_sq_expression(l::Integer, z2::AbstractString)
    l == 0 && return "1"
    chi = join(("$c * $z2^$k" for (k, c) in enumerate(_BLATT_WEISSKOPF_CHI[l][2:end])), " + ")
    return "$z2^$l / ($(_BLATT_WEISSKOPF_CHI[l][1]) + $chi)"
end

"""
    tfpwa_multichannel_bw(m, channels)

TF-PWA multichannel Breit–Wigner (`BWR_LS` with `fix_bug1 = false`): each channel
`(; gsq, ma, mb, l, d)` contributes `gsq * σ * 2p/√σ * F_l²(d p)` to `m Γ`, a factor `σ`
more than HadronicLineshapes' `MultichannelBreitWigner`. With `2p·√σ = λ^(1/2)(σ, ma², mb²)`
and `(d p)² = d² λ / (4σ)` it needs no auxiliary functions.
"""
function tfpwa_multichannel_bw(m, channels)
    terms = map(channels) do c
        ma, mb, d = _expr_number(c.ma), _expr_number(c.mb), _expr_number(c.d)
        λ = "(sigma - ($ma + $mb)^2) * (sigma - ($ma - $mb)^2)"
        z2 = "($d^2 * $λ / (4 * sigma))"
        "$(_expr_number(c.gsq)) * sqrt($λ) * $(_blatt_weisskopf_sq_expression(c.l, z2))"
    end
    return GenericFunction("1 / ($(_expr_number(m))^2 - sigma - i * (" * join(terms, " + ") * "))")
end

# Nonresonant exponential exp(-(α + iβ)(σ - m0²)); TF-PWA's charge sign of `C(New)`
# lives in `MAGIC_SIGNS`, like for every other C-dependent resonance
function nr_exp_lineshape()
    α, β = param_real("NR(0-)SPp_alpha"), param_real("NR(0-)SPp_beta")
    m0 = nominal_mass["NR(0-)SPp"]
    return GenericFunction("exp(-($(_expr_number(α)) + $(_expr_number(β)) * i) * (sigma - $(_expr_number(m0))^2))")
end

"""
    dstar_delta_lineshape()

The D* line as a delta distribution at m = m(D*): TF-PWA fixes the D* mass
(narrow-width approximation), so the model is defined only at m(D0 pi) = m. As a
lineshape of the invariant mass squared `σ` of the D* node,

    g(σ) = exp(-0.5 * ((√σ - m) / w)^n),    g(m²) = 1,

a flat-top profile of half-width ≈ w (`n` even). Integrals over the full four-body phase
space reproduce the fixed-mass ones up to O(w²); near m the profile is flat to
O((Δm/w)^n), so inputs whose m(D0 pi) misses m by much less than w (e.g. rounding of
the four-vectors) evaluate to exactly the fixed-mass amplitude. The minus sign belongs
to the literal "-0.5", so the value does not depend on how a reader binds unary minus
and `^`.
"""
dstar_delta_lineshape() = GenericFunction(
    "exp(-0.5 * ((sqrt(sigma) - $(_expr_number(nominal_mass["Dst"]))) / $(_expr_number(DSTAR_DELTA_WIDTH)))^$(DSTAR_DELTA_ORDER))")

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
    return tfpwa_multichannel_bw(nominal_mass[name], channels)
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

# ---------------------------------------------------------------------------
# Resonance table: chains, spins, LS couplings, lineshapes
# ---------------------------------------------------------------------------

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
        coupling_keys=total_x3872, lineshape=:adhoc_q0_bwr_ls_l0,
    ))
    push!(rows, (
        resonance_name="X(3872)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
        coupling_keys=(total_x3872..., "X(3872)->Dst.D_g_ls_1"),
        lineshape=:adhoc_q0_bwr_ls_l2,
    ))
    push!(rows, (
        resonance_name="X(3915)(0-)", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
    ))
    push!(rows, (
        resonance_name="chi(c2)(3930)", topology=:DxD,
        propagator_two_j=4, root_two_ls=(4, 4), daughter_two_ls=(4, 2),
        coupling_keys=("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l2,
    ))
    for name in ("X(3940)(1.)", "X(3993)", "X(4300)")
        total = (production_coupling_key(name),)
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
            coupling_keys=total, lineshape=:bwr_ls_l0,
        ))
        push!(rows, (
            resonance_name=name, topology=:DxD,
            propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(4, 2),
            coupling_keys=(total..., "$(name)->Dst.D_g_ls_1"),
            lineshape=:bwr_ls_l2,
        ))
    end
    push!(rows, (
        resonance_name="Psi(4040)", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0",),
        lineshape=:bwr_l1,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPp", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0",),
        lineshape=:nr_exp,
    ))
    push!(rows, (
        resonance_name="NR(1.)PSp", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(0, 2),
        coupling_keys=("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="NR(0-)SPm", topology=:DxD,
        propagator_two_j=0, root_two_ls=(0, 0), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="NR(1-)PPm", topology=:DxD,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=(2, 2),
        coupling_keys=("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0",),
        lineshape=:constant,
    ))
    push!(rows, (
        resonance_name="X0(2900)", topology=:dk,
        propagator_two_j=0, root_two_ls=(2, 2), daughter_two_ls=(0, 0),
        coupling_keys=("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0",),
        lineshape=:x2900_bwr_l0,
    ))
    total_x1 = ("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0",)
    daughter_x1 = (2, 0)
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(0, 0), daughter_two_ls=daughter_x1,
        coupling_keys=total_x1, lineshape=:x2900_bwr_l1,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(2, 2), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_1"),
        lineshape=:x2900_bwr_l1,
    ))
    push!(rows, (
        resonance_name="X1(2900)", topology=:dk,
        propagator_two_j=2, root_two_ls=(4, 4), daughter_two_ls=daughter_x1,
        coupling_keys=(total_x1..., "Bp->X1(2900).Dst_g_ls_2"),
        lineshape=:x2900_bwr_l1,
    ))
    return DataFrame(rows)
end

const resonance_chains_df_raw = build_resonance_chains_df()

# ---------------------------------------------------------------------------
# Matching factors to TF-PWA
# ---------------------------------------------------------------------------

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
#   N_propagator_spin  — propagator_two_j (CascadeDecays spin norm; D* line fixed at jp"1+")
#
# Particle-2 convention
# ---------------------
# The Jacob–Wick particle-2 convention has two halves: the phase (-1)^{(j₂-λ₂)/2}
# in the two-body coupling, and the π-rotated helicity frame when descending into
# a child-2 line.  CascadeDecays applies both from v0.4.0 on; before, `RecouplingLS`
# carried the phase while the frame descent ignored it, and `MissingParticleTwoPhaseLS`
# cancelled the phase to restore consistency.  TF-PWA omits the convention entirely;
# see `docs/tfpwa_review/tfpwa_modelling_issues.qmd`.
#
# Applied consistently the two halves combine into a per-chain *constant*, so no
# recoupling workaround is needed — the whole effect lives in MAGIC_SIGNS below.

const MAGIC_SIGNS = Dict{String,Float64}(
    "X(3872)" => -1.0,
    "X(3915)(0-)" => -1.0,
    "chi(c2)(3930)" => -1.0,
    "X(3940)(1.)" => 1.0,
    "X(3993)" => -1.0,
    "Psi(4040)" => 1.0,
    "X(4300)" => 1.0,
    # TF-PWA `C(New)` with `C: 1` (C- sign)
    "NR(0-)SPp" => -1.0,
    "NR(1.)PSp" => -1.0,
    "NR(0-)SPm" => 1.0,
    "NR(1-)PPm" => 1.0,
    "X0(2900)" => 1.0,
    # Flipped when moving to CascadeDecays v0.4.0.  The upgrade multiplies every
    # X1(2900) chain by exactly -1 and leaves the other 16 chains untouched (a
    # pure per-chain sign, no event dependence); (-1)^{j_X} is the expected size
    # for a spin-1 DK resonance entered as child 2: TF-PWA omits the Jacob-Wick
    # particle-2 convention, which for this chain reduces to this constant.  All other
    # signs are the C- signs of TF-PWA's `C(...)` wrappers (`C: 1`).
    # See archive/notes/note-cascadedecays-v040.md.
    "X1(2900)" => -1.0,
)
@assert Set(keys(MAGIC_SIGNS)) == Set(all_resonance_names)

const C_DEPENDENT_RESONANCES = Set([
    "X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3993)", "NR(0-)SPp", "NR(1.)PSp"
])

const CPLUS_EXCLUDED_RESONANCES = Set([
    "X0(2900)", "X1(2900)"
])

function magic_sign(resonance_name::String, charge::Symbol = :Cminus)
    base = MAGIC_SIGNS[resonance_name]
    if charge in (:Cplus, :cplus, :+)
        if resonance_name in CPLUS_EXCLUDED_RESONANCES
            return 0.0
        elseif resonance_name in C_DEPENDENT_RESONANCES
            return -base
        end
    end
    return base
end

const DSTAR_PROPAGATOR_TWO_J = 2  # jp"1+" on D* (Dst) line (1, 2) in every chain

vertex_l(two_ls) = div(two_ls[1], 2)

vertex_blatt_l(r::RecouplingLS) = vertex_l(r.two_ls)

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

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

const external_spins = SystemSpins(0, 0, 0, 0; two_h0=0)
const dxd_topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))
const kinematic_task = KinematicTask((dxd_topology, dk_topology))

function event_point(row)
    pDminus = FourVector(row.Dm_px, row.Dm_py, row.Dm_pz; E=row.Dm_E)
    pD0 = FourVector(row.D0_px, row.D0_py, row.D0_pz; E=row.D0_E)
    pKplus = FourVector(row.Kp_px, row.Kp_py, row.Kp_pz; E=row.Kp_E)
    piplus = FourVector(row.pip_px, row.pip_py, row.pip_pz; E=row.pip_E)
    return KinematicPoint(kinematic_task, (pD0, piplus, pDminus, pKplus))
end

# ---------------------------------------------------------------------------
# Model: chains and cascades
# ---------------------------------------------------------------------------

function enrich_resonance_chains_df!(df)
    df.coupling_value = resolve_coupling_keys.(df.coupling_keys)
    df.matching_factor = chain_matching_factor.(eachrow(df))
    return df
end

function build_dxd_chain(lineshape, two_j, production::Recoupling, decay::Recoupling; resonance_name::String = "")
    res_ls = isempty(resonance_name) ? lineshape : NamedLineshape("lineshape_$(resonance_name)", lineshape)
    dst_ls = NamedLineshape("lineshape_Dst", dstar_delta_lineshape())
    return DecayChain(
        dxd_topology,
        external_spins;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), dst_ls),
            ((1, 2), 3) => Propagator(two_j, res_ls),
        ),
        vertices=(
            (((1, 2), 3), 4) => Vertex(production, BlattWeisskopf{vertex_blatt_l(production)}(WELL_SIZE)),
            ((1, 2), 3) => Vertex(decay, BlattWeisskopf{vertex_blatt_l(decay)}(WELL_SIZE)),
            (1, 2) => Vertex(RecouplingLS((2, 0))),
        ),
    )
end

function build_dk_chain(lineshape, two_j, root::Recoupling, daughter::Recoupling; resonance_name::String = "")
    res_ls = isempty(resonance_name) ? lineshape : NamedLineshape("lineshape_$(resonance_name)", lineshape)
    dst_ls = NamedLineshape("lineshape_Dst", dstar_delta_lineshape())
    return DecayChain(
        dk_topology,
        external_spins;
        propagators=(
            (1, 2) => Propagator(ThreeBodyDecays.str2jp("1+"), dst_ls),
            (3, 4) => Propagator(two_j, res_ls),
        ),
        vertices=(
            ((1, 2), (3, 4)) => Vertex(root, BlattWeisskopf{vertex_blatt_l(root)}(WELL_SIZE)),
            (3, 4) => Vertex(daughter, BlattWeisskopf{vertex_blatt_l(daughter)}(WELL_SIZE)),
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
    root = RecouplingLS(row.root_two_ls)
    daughter = RecouplingLS(row.daughter_two_ls)
    if row.topology == :dk
        return build_dk_chain(lineshape, row.propagator_two_j, root, daughter; resonance_name = row.resonance_name)
    end
    return build_dxd_chain(lineshape, row.propagator_two_j, root, daughter; resonance_name = row.resonance_name)
end


# <resonance>_l<production l>_l<resonance decay l>, e.g. "X(3940)(1.)_l1_l0"
function chain_name(resonance_name::String, row)
    "$(resonance_name)_l$(vertex_l(row.root_two_ls))_l$(vertex_l(row.daughter_two_ls))"
end

function resonance_chain_names(resonance_name::String)
    [chain_name(resonance_name, row) for row in eachrow(resonance_chain_rows(resonance_name))]
end

function build_resonance_cascade(resonance_name::String; charge::Symbol = :Cminus)
    if (charge in (:Cplus, :cplus, :+)) && (resonance_name in CPLUS_EXCLUDED_RESONANCES)
        return nothing
    end
    rows = collect(eachrow(resonance_chain_rows(resonance_name)))
    chains = Tuple(build_chain_from_row(row) for row in rows)
    sign_factor = ((charge in (:Cplus, :cplus, :+)) && (resonance_name in C_DEPENDENT_RESONANCES)) ? -1.0 : 1.0
    effective_couplings = Tuple(
        rows[i].coupling_value * rows[i].matching_factor * sign_factor
        for i in eachindex(rows)
    )
    names = Tuple(chain_name(resonance_name, row) for row in rows)
    return CascadeDecay(
        chains,
        dxd_topology;
        couplings=effective_couplings,
        names=names,
    )
end

function build_all_resonance_cascade(names=all_resonance_names; charge::Symbol = :Cminus)
    filtered = (charge in (:Cplus, :cplus, :+)) ? filter(n -> !(n in CPLUS_EXCLUDED_RESONANCES), names) : names
    cascades = [build_resonance_cascade(name; charge) for name in filtered]
    merge(cascades...)
end
