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
#
# Particle-2 phase (Issue A, not part of M_chain)
# ----------------------------------------------
# CascadeDecays applies the Jacob–Wick factor (-1)^{(j₂-λ₂)/2} at every two-body
# vertex.  RecouplingLS builds H_{λ₁,λ₂} with the standard ⟨j₁,λ₁; j₂,-λ₂|…⟩
# CG convention.  TF-PWA omits the particle-2 phase; see
# `docs/tfpwa_review/tfpwa_modelling_issues.qmd`.
#
# For integer-spin mesons, (-1)^{j₂-λ₂} = (-1)^{j₂+λ₂} because (-1)^{2λ₂}=1, so
# the phase is *invariant* under λ₂ → -λ₂.  It distinguishes λ₂=0 from λ₂=±1,
# not +1 from -1.  When only λ₂=±1 contribute at the dk root, the phase is a
# common +1 and no recoupling workaround is needed (X₁(2900) L=1).  When the
# root LS coupling is nonzero at λ₂=0, λ₂=0 and λ₂=±1 enter with opposite signs
# and the mismatch is helicity dependent — it cannot be absorbed into the scalar
# M_chain above.  Set `root_recoupling` on each row in `resonance_table.jl`.

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
vertex_blatt_l(r::MissingParticleTwoPhaseLS) = vertex_l(r.two_ls)

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
