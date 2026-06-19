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
