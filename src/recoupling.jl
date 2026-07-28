"""
    MissingParticleTwoPhaseLS(two_ls)

Workaround recoupling that applies the Jacob–Wick particle-2 phase a second time so
it cancels CascadeDecays' built-in factor ``(-1)^{(j_2-\\lambda_2)/2}`` in
`_vertex_coupling_value`. TF-PWA **omits** this phase (it is missing from that
implementation, not a harmless convention difference).

Use this at the dk root when both ``\\lambda_2 = 0`` and ``\\lambda_2 = \\pm 1``
helicity routes contribute: the phase weights those routes differently and cannot
be folded into a scalar [`chain_matching_factor`](@ref).
"""
struct MissingParticleTwoPhaseLS <: Recoupling
    two_ls::Tuple{Int,Int}
end

function ThreeBodyDecays.amplitude(cs::MissingParticleTwoPhaseLS, helicities, spins)
    _, _, two_j2 = spins
    _, two_lambda2 = helicities
    exponent_num = two_j2 - two_lambda2
    iseven(exponent_num) || error("particle-2 phase requires two_j2 - two_lambda2 to be even")
    phase = isodd(div(exponent_num, 2)) ? -1 : 1
    return phase * ThreeBodyDecays.amplitude(RecouplingLS(cs.two_ls), helicities, spins)
end

"""
    root_recoupling(two_ls, kind)

Root-vertex recoupling selector.  `kind` is set per row in `resonance_table.jl`:

  * `:standard` — [`RecouplingLS`](@ref)
  * `:missing_particle2` — [`MissingParticleTwoPhaseLS`](@ref) (TF-PWA omits the phase)

Used when building chains from `resonance_table.jl`.  Charmonium (`:DxD`) rows use
`:standard`; only `:dk` rows need `:missing_particle2` today.
"""
function root_recoupling(two_ls, kind::Symbol)
    kind === :standard && return RecouplingLS(two_ls)
    kind === :missing_particle2 && return MissingParticleTwoPhaseLS(two_ls)
    error("invalid root_recoupling=$(repr(kind)); use :standard or :missing_particle2")
end
