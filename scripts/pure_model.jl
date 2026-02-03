# Numbering: mD, mK, mDx as 1,2,3
# It is circular permutation of Dx D K

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
# 
using ThreeBodyDecays.PartialWaveFunctions
using HadronicLineshapes
using OrderedCollections
using ThreeBodyDecays
using LinearAlgebra
using Measurements
using Statistics
using Parameters
using DataFrames.PrettyTables
using DataFrames
using QuadGK
using Plots
using YAML
using JSON
# 
using Random
Random.seed!(1234)

theme(:boxed)


# ==================================================================
# 
#  C O D E   F U N C T I O N S   A N D   L I N E S H A P E S
#
# ==================================================================

function BreitWignerSDwaves(; m, Γ, γS, d)
    fr = [γS, 1 - γS]
    channels = map(zip([0, 2], fr)) do (l, x)
        ma, mb = mD, mDx
        # 
        p = HadronicLineshapes.breakup(m, ma, mb)
        gsq = m * Γ / (2 * p / m) * x / BlattWeisskopf{l}(d)(p)^2
        # 
        (; gsq, ma, mb, l, d)
    end
    MultichannelBreitWigner(; m, channels)
end

begin #
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))
    # 
    const ConstantLineshape = WrapFlexFunction(x -> 1.0)
end




# matching factors

function nominal_mass(any, m_min, m_max)
    @warn "Type is not recognized return middle of phsp!\n Type: $(typeof(any)) -- Not BreitWigner, not MultichannelBreitWigner"
    return (m_max + m_min) / 2
end
function nominal_mass(X::BW, m_min, m_max) where BW<:Union{BreitWigner,MultichannelBreitWigner}
    mR = X.m
    m_min < mR < m_max && return mR
    @warn "mR is beyond phase space, mR = $(round(mR; digits=2)), not in $(round.((m_min, m_max); digits=2))"
    return (m_max + m_min) / 2
end


function matching_factor(any)
    @warn "Type is not recognized for the matching_factor factor. Assume 1.0"
    return 1.0
end

function matching_factor(H::VertexFunction{RecouplingLS,BlattWeisskopf{N}}) where {N}
    julia = H.ff.d^orbital_momentum(H.ff)
    return 1 / julia
end

function matching_factor(ch::DecayChain)
    (; tbs, two_j, k, Xlineshape, HRk, Hij) = ch
    (; ms) = tbs
    ms² = ms^2
    # 
    mv = sqrt.(lims(ms; k))
    mR = nominal_mass(Xlineshape, mv...)
    i, j = ij_from_k(k)
    _ff_Rk_0 = HRk.ff(ms²[4], mR^2, ms²[k])
    _ff_ij_0 = Hij.ff(mR^2, ms²[i], ms²[j])
    # _fX = matching_factor(Xlineshape)
    _fVRk = _ff_Rk_0 * (HRk.h.two_ls[1] != 0 ? matching_factor(HRk) : 1.0)
    _fVij = _ff_ij_0 * (HRk.h.two_ls[1] != 0 ? matching_factor(Hij) : 1.0)

    return 1 / sqrt(two_j + 1) * _fVRk * _fVij
end




# compute full amplitude

struct DalitzAndDecay{T}
    σs::MandelstamTuple{T}
    cosθ::T
    ϕ::T
end

function ThreeBodyDecays.amplitude(three_body_model::ThreeBodyDecay, dd::DalitzAndDecay; refζs=(1, 1, 2, 1)) # default value for refζs -- how Dx helicity frame angles are computed
    @unpack σs, cosθ, ϕ = dd
    total_amp = 0.0
    jDx = 1
    _O = amplitude(three_body_model, σs; refζs) # order: -1,0,1
    _D = [wignerD(jDx, λ, 0, ϕ, cosθ, 0.0) for λ in -1:1] .|> conj # order: -1,0,1
    total_amp = sum(reshape(_O, 3) .* _D)
    return total_amp
end




# ==================================================================
# 
#  P R O C E D U R A L   P R O C E S S I N G
#
# ==================================================================



params = let
    json_path = joinpath(@__DIR__, "..", "data", "final_params.json")
    JSON.parsefile(json_path)["value"]
end

config = YAML.load_file(joinpath(@__DIR__, "..", "data", "config_c.yml"))



begin
    const mB = config["particle"]["\$top"]["Bp"]["mass"]
    # const mD0 = config["particle"]["\$finals"]["D0"]["mass"]
    const mD = config["particle"]["\$finals"]["D"]["mass"]
    const mDx = config["particle"]["Dst"]["mass"]
    const mK = config["particle"]["\$finals"]["K"]["mass"]
end;

(two_js, pc), (_, pv) = map(["0+", "0-"]) do jp0
    ThreeBodySpinParities("0-", "0-", "1-"; jp0)
end;

tbs = let
    ms = ThreeBodyMasses(mD, mK, mDx; m0=mB)
    ThreeBodySystem(ms, two_js)
end;


const standard_d = 3.0;

const EFF = BreitWigner(3.85, 0.001);

BW_4040 = let
    m = params["Psi(4040)_mass"]
    Γ = params["Psi(4040)_width"]
    BreitWigner(m, Γ, mDx, mD, 1, standard_d)
end

BW_4000 = let
    m = params["X(3940)(1.)_mass"]
    Γ = params["X(3940)(1.)_width"]
    θ = params["X(3940)(1.)_theta0"]
    BreitWignerSDwaves(; m, Γ, γS=cos(θ)^2, d=standard_d)
end

BW_4010 = let
    m = params["X(3993)_mass"]
    Γ = params["X(3993)_width"]
    θ = params["X(3993)_theta0"]
    BreitWignerSDwaves(; m, Γ, γS=cos(θ)^2, d=standard_d)
end

BW_4300 = let
    m = params["X(4300)_mass"]
    Γ = params["X(4300)_width"]
    θ = params["X(4300)_theta0"]
    BreitWignerSDwaves(; m, Γ, γS=cos(θ)^2, d=standard_d)
end


BW_Tcs0 = let
    l = 0
    m = params["X0(2900)_mass"]
    Γ = params["X0(2900)_width"]
    BreitWigner(m, Γ, mD, mK, l, standard_d)
end
BW_Tcs1 = let
    l = 1
    m = params["X1(2900)_mass"]
    Γ = params["X1(2900)_width"]
    BreitWigner(m, Γ, mD, mK, l, standard_d)
end

BW_χ2_3930 = let
    l = 2 # D-wave
    m = params["chi(c2)(3930)_mass"]
    Γ = params["chi(c2)(3930)_width"]
    BreitWigner(m, Γ) # no energy dep, BRW no LS in the config_c
end

# resonances

resonances =
    [
        (; jp=jp"1+", name="EFF(1++)", lineshape=EFF), (; jp=jp"0-", name="ηc(3945)", lineshape=BreitWigner(3.945, 0.13)),
        (; jp=jp"2+", name="χc2(3930)", lineshape=BW_χ2_3930),
        (; jp=jp"1+", name="hc(4000)", lineshape=BW_4000),
        (; jp=jp"1+", name="χc1(4010)", lineshape=BW_4010),
        (; jp=jp"1-", name="ψ(4040)", lineshape=x -> 1), #BW_4040
        (; jp=jp"1+", name="hc(4300)", lineshape=BW_4300),
        # Tcbarsbar
        (; jp=jp"0+", name="Tcs0(2870)", lineshape=BW_Tcs0),
        (; jp=jp"1-", name="Tcs1(2900)", lineshape=BW_Tcs1),
        # NR
        (; jp=jp"1-", name="NR(1--)", lineshape=ConstantLineshape),
        (; jp=jp"0-", name="NR(0--)", lineshape=ConstantLineshape),
        (; jp=jp"1+", name="NR(1++)", lineshape=ConstantLineshape),
        (; jp=jp"0-", name="NR(0-+)", lineshape=NRexp(αβ=0.11 - 0.34im, m0=4.35)),
    ] |> DataFrame
# 
decay_chains = [
    (k=2, resonance_name="EFF(1++)", l=0),
    (k=2, resonance_name="EFF(1++)", l=2),
    # 
    (k=2, resonance_name="ηc(3945)"),
    (k=2, resonance_name="χc2(3930)"),
    (k=2, resonance_name="hc(4000)", l=0),
    (k=2, resonance_name="hc(4000)", l=2),
    (k=2, resonance_name="χc1(4010)", l=0),
    (k=2, resonance_name="χc1(4010)", l=2),
    (k=2, resonance_name="ψ(4040)"),
    (k=2, resonance_name="hc(4300)", l=0),
    (k=2, resonance_name="hc(4300)", l=2),
    # 
    (k=2, resonance_name="NR(1--)"),
    (k=2, resonance_name="NR(0--)"),
    (k=2, resonance_name="NR(1++)", l=0),
    (k=2, resonance_name="NR(0-+)"),
    # 
    (k=3, resonance_name="Tcs0(2870)"),
    (k=3, resonance_name="Tcs1(2900)", L=0),
    (k=3, resonance_name="Tcs1(2900)", L=1),
    (k=3, resonance_name="Tcs1(2900)", L=2),
];

chains = let
    resonance_dict = LittleDict(
        resonances.name .=> NamedTuple.(eachrow(resonances)))
    #
    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k) # takes into account l and L from named tuple
        @unpack L, S, l, s = comprete_data
        # 
        two_j = _jp.two_j
        # 
        d = standard_d
        Xlineshape = resonance_dict[resonance_name].lineshape
        # 
        HRk = VertexFunction(RecouplingLS((L, S) .|> x2), BlattWeisskopf{div(x2(L), 2)}(d))
        Hij = VertexFunction(RecouplingLS((l, s) .|> x2), BlattWeisskopf{div(x2(l), 2)}(d))
        # 
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* [ch.Hij.h.two_ls[1] |> d2 for ch in chains]
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.h.two_ls[1] |> d2)" : "" for ch in chains]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains)), chains))
end;




# testing: use the DPD configuration from data/crosscheck_event.json


dalitz_dpd = let

    json_path = joinpath(@__DIR__, "..", "data", "crosscheck_event.json")
    event = JSON.parsefile(json_path)
    dpd = event["dpd_kinematics"]

    # Map DPD invariants to the Mandelstam variables for (D, K, Dx)
    # Convention in ThreeBodyDecays: σ₁ = m²(23), σ₂ = m²(13), σ₃ = m²(12)
    # Here: 1 ≡ D, 2 ≡ K, 3 ≡ Dx
    σs_dpd = (
        σ1=dpd["msq_KDx"],  # m²(K, Dx)  → pair (2,3)
        σ2=dpd["msq_DxD"],  # m²(D, Dx)  → pair (1,3)
        σ3=dpd["msq_DK"],   # m²(D, K)   → pair (1,2)
    )

    DalitzAndDecay(
        σs_dpd,
        dpd["cos_theta_D_in_Dx"],
        dpd["phi_D_in_Dx"],
    )
end

full_amplitude = amplitude(model_pure, dalitz_dpd; refζs=(1, 1, 2, 2))

amplitude(model_pure[9], dalitz_dpd; refζs=(1, 1, 2, 1))



println("## Amplitude at DPD cross-check event")
println("DalitzAndDecay = ", dalitz_dpd)
println("amplitude(model_pure, dalitz_dpd) = ", full_amplitude)

# amplitudes for individual decay chains
println("\n## Amplitudes per decay chain at DPD cross-check event")
chain_amps = [
    let
        chain_name = name
        chain = model_pure.chains[i]
        (chain_name, chain, amplitude=amplitude(model_pure[i], dalitz_dpd))
    end
    for (i, name) in enumerate(model_pure.names)
]

df_chain_amps = DataFrame(chain_amps)

@assert df_chain_amps.amplitude |> sum ≈ full_amplitude atol = 1e-10

transform!(df_chain_amps, :chain => ByRow() do ch
    matching_factor(ch)
end => :matching_factor)

transform!(df_chain_amps, [:amplitude, :matching_factor] => ByRow() do a, m
    a * m
end => :A_x_m)
