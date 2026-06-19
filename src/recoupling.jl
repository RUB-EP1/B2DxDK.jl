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
