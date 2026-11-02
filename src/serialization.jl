# amplitude-serialization of the analysis-specific lineshapes. None of them is a
# standard function type, so all are written as `generic_function`: an expression in
# the single ASCII variable `sigma` (invariant mass squared of the propagator node).
# Masses enter as written in the model ("4.35^2", "(2.01026 + 1.86965)^2"), so every
# number can be traced to its parameter; derived couplings are inlined as numbers.

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
    tfpwa_bwls_expression(ls::TFPWAMultichannelBreitWigner)

TF-PWA multichannel Breit–Wigner as a closed expression in `sigma`. Each channel
contributes `gsq * σ * 2p/√σ * F_l²(d p)` to `m Γ`; with
`2p·√σ = λ^(1/2)(σ, ma², mb²)` and `(d p)² = d² λ / (4σ)` this needs no auxiliary
functions.
"""
function tfpwa_bwls_expression(ls::TFPWAMultichannelBreitWigner)
    terms = map(ls.channels) do c
        ma, mb, d = _expr_number(c.ma), _expr_number(c.mb), _expr_number(c.d)
        λ = "(sigma - ($ma + $mb)^2) * (sigma - ($ma - $mb)^2)"
        z2 = "($d^2 * $λ / (4 * sigma))"
        "$(_expr_number(c.gsq)) * sqrt($λ) * $(_blatt_weisskopf_sq_expression(c.l, z2))"
    end
    return "1 / ($(_expr_number(ls.m))^2 - sigma - i * (" * join(terms, " + ") * "))"
end

nr_exp_expression(ls::NRExpLineshape) =
    "exp(-($(_expr_number(real(ls.αβ))) + $(_expr_number(imag(ls.αβ))) * i) * (sigma - $(_expr_number(ls.m0))^2))"

# "-0.5 * x^n": the minus belongs to the literal, so the value does not depend on whether a
# reader binds unary minus before or after ^
dstar_delta_expression(ls::DstarDeltaLineshape) =
    "exp(-0.5 * ((sqrt(sigma) - $(_expr_number(ls.m))) / $(_expr_number(ls.w)))^$(ls.n))"

_generic_function(expression) =
    (CascadeDecaysIO.LittleDict{String,Any}("type" => "generic_function", "expression" => expression), Dict{String,Any}())

CascadeDecaysIO.serializeToDict(ls::TFPWAMultichannelBreitWigner) = _generic_function(tfpwa_bwls_expression(ls))
CascadeDecaysIO.serializeToDict(ls::NRExpLineshape) = _generic_function(nr_exp_expression(ls))
CascadeDecaysIO.serializeToDict(ls::DstarDeltaLineshape) = _generic_function(dstar_delta_expression(ls))
