# Matching is evaluated per chain (one row of `resonance_chains_df`).  Total
# matching is the product of vertex and lineshape contributions:
#
#     M_chain = M_sign × M_particle2 × M_vertex × M_lineshape / N_propagator_spin
#
# Dependencies:
#   M_vertex           — topology, resonance_name, root_two_ls, daughter_two_ls, lineshape
#                        (lineshape sets decay reference mass for adhoc-q0 DxD chains)
#   M_lineshape        — resonance_name, lineshape (multichannel γ₀/γ₂ split)
#   M_sign             — `MAGIC_SIGNS[resonance_name]` (TF-PWA overall sign)
#   M_particle2        — particle-2 convention constant (see below)
#   N_propagator_spin  — propagator_two_j (CascadeDecays spin norm; D* line fixed at jp"1+")
#
# Particle-2 convention
# ---------------------
# The Jacob–Wick particle-2 convention has two halves: the phase (-1)^{(j₂-λ₂)/2}
# in the two-body coupling, and the π-rotated helicity frame when descending into
# a child-2 line.  CascadeDecays applies both from v0.4.0 on (before, `RecouplingLS`
# carried the phase while the frame descent ignored it — see
# `docs/note-cascadedecays-v040.md`).  TF-PWA omits the convention entirely; see
# `docs/tfpwa_review/tfpwa_modelling_issues.qmd`.
#
# Applied consistently the two halves combine into a per-chain *constant*
#
#     A_CascadeDecays = (-1)^{j₂} A_TFPWA,
#
# with j₂ the spin of the child-2 line that is entered.  In the DxD topology the
# D* is child 1 at every vertex and every child 2 (K⁺, D⁻, π⁺) is spinless, so the
# factor is +1.  In the dk topology the DK resonance is child 2 of the root and it
# decays, so the factor is (-1)^{j_X}: +1 for X₀(2900), −1 for X₁(2900).  Being a
# constant it belongs in M_chain; no recoupling workaround is needed.

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

vertex_blatt_l(r::RecouplingLS) = vertex_l(r.two_ls)

"""
    chain_particle2_convention_factor(row)

`(-1)^{j₂}` over the child-2 lines that the chain descends into — the constant
relating the CascadeDecays particle-2 convention to the TF-PWA one (see the note
at the top of this file).  DxD chains only ever descend into spinless child-2
lines, so they carry `+1`; dk chains descend into the DK resonance and carry
`(-1)^{j_X}`.
"""
function chain_particle2_convention_factor(row)
    row.topology === :dk || return 1.0
    return isodd(div(row.propagator_two_j, 2)) ? -1.0 : 1.0
end

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
        chain_particle2_convention_factor(row) *
        chain_vertex_matching_factor(row) *
        chain_lineshape_matching_factor(row.resonance_name, row.lineshape) /
        chain_propagator_spin_norm(row)
    )
end
