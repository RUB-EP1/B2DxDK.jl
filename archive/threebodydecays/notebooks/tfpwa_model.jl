### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ db4f869a-fac9-11f0-313e-7b298f5edaf2
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec("ThreeBodyDecays")
        Pkg.PackageSpec("Plots")
        Pkg.PackageSpec(url="https://github.com/mmikhasenko/InstructionalDecayTrees.jl.git")
        Pkg.PackageSpec(url="https://github.com/mmikhasenko/FourVectors.jl.git")
        Pkg.PackageSpec("PartialWaveFunctions")
        Pkg.PackageSpec("HadronicLineshapes")
        Pkg.PackageSpec("JSON")
        Pkg.PackageSpec("YAML")
    ])
    # 
    using YAML
    using JSON
    using Plots
    using FourVectors
    using ThreeBodyDecays
    using HadronicLineshapes
    using PartialWaveFunctions
    using InstructionalDecayTrees
end

# ╔═╡ f15f6b55-11a4-4b95-9c65-dd324925592e
md"""
# Cascade decay of B

This document describes the amplitude computation for the cascade decay of a $B$ meson implemented in `tfpwa_model.jl`.

## Decay Chain

The decay proceeds through the following cascade:

$$B \to \psi K$$

where $\psi$ subsequently decays as:

$$\psi \to D^* \bar{D}$$

and $D^*$ decays as:

$$D^* \to D^0 \pi$$

The full decay chain is: $B \to \psi K \to (D^* \bar{D}) K \to (D^0 \pi) \bar{D} K$.

## Kinematic Variables

The decay is described using invariant masses and helicity angles:

- **Invariant masses:**
  - variable $m_B^2 = (p_1 + p_2 + p_3 + p_4)^2$ - parent $B$ meson
  - variable $m_\psi^2 = (p_1 + p_2 + p_3)^2$ - intermediate $\psi$ resonance
  - variable $m_{D^*}^2 = (p_1 + p_2)^2$ - intermediate $D^*$ resonance
  - Individual particle masses: $m_1^2 = p_1^2$, $m_2^2 = p_2^2$, $m_3^2 = p_3^2$, $m_4^2 = p_4^2$

- **Helicity angles:**
  - variable $(\cos\theta_B, \phi_B)$ - angles of the $(1,2,3)$ system in the $B$ rest frame
  - variable $(\cos\theta_\psi, \phi_\psi)$ - angles of the $(1,2)$ system in the $\psi$ rest frame
  - variable $(\cos\theta_{D^*}, \phi_{D^*})$ - angles of particle 1 in the $D^*$ rest frame

## Two-Body Decay Amplitude

For a two-body decay $0 \to 1 + 2$, the amplitude in the helicity basis is:

$$\mathcal{A}_{\lambda_1 \lambda_2}^{\lambda_0} = \sqrt{2j_0 + 1} \, D_{\lambda_0,\lambda_1-\lambda_2}^{j_0*}(\phi, \cos\theta, 0) \, \mathcal{A}_{\text{rec}} \, F(m_0^2, m_1^2, m_2^2)$$

where:
- variable $j_0$ is the spin of the parent particle (in units of $\hbar$)
- variable $\lambda_0, \lambda_1, \lambda_2$ are the helicities of particles 0, 1, and 2
- variable $D_{\lambda,\Delta\lambda}^{j*}(\phi, \cos\theta, 0)$ is the complex conjugate of the Wigner $D$-function
- variable $\mathcal{A}_{\text{rec}}$ is the recoupling amplitude (LS coupling coefficient)
- variable $F(m_0^2, m_1^2, m_2^2)$ is the form factor (Blatt-Weisskopf barrier factor)

### Wigner D-Function

The Wigner $D$-function describes the rotation from the helicity frame to the lab frame:

$$D_{\lambda,\Delta\lambda}^{j}(\phi, \cos\theta, 0) = d_{\lambda,\Delta\lambda}^{j}(\cos\theta) \, e^{i\lambda\phi}$$

where $d_{\lambda,\Delta\lambda}^{j}(\cos\theta)$ is the Wigner small $d$-function.

### Recoupling Amplitude

The recoupling amplitude $\mathcal{A}_{\text{rec}}$ connects the helicity basis to the LS coupling basis:

$$\mathcal{A}_{\text{rec}} = \sqrt{\frac{2l+1}{2j_0+1}} \, \langle j_1, \lambda_1; j_2, -\lambda_2 | s, \Delta\lambda \rangle \, \langle l, 0; s, \Delta\lambda | j_0, \Delta\lambda \rangle$$

where:
- variables $l$ and $s$ are the orbital angular momentum and total spin in the LS coupling scheme
- variable $\Delta\lambda = \lambda_1 - \lambda_2$ is the helicity difference
- The Clebsch-Gordan coefficients couple the daughter spins to the total spin $s$, and then couple the orbital and spin angular momenta to the parent spin $j_0$

### Form Factor

The Blatt-Weisskopf form factor accounts for the angular momentum barrier:

$$F_l(q^2) = \frac{1}{\sqrt{1 + (q R)^2}}$$

for $l=0$, and more complex expressions for higher $l$, where:
- variable $q$ is the breakup momentum
- variable $R = d_0$ is the interaction radius (typically $d_0 = 3$ GeV$^{-1}$)

## Cascade Amplitude

For the full cascade decay, the amplitude is constructed by summing over intermediate helicities:

$$\mathcal{A} = \sum_{\lambda_\psi = -1}^{1} \sum_{\lambda_{D^*}=-1}^{1} \mathcal{A}_{B,\lambda_\psi 0}^{\lambda_B} \, \mathcal{A}_{\psi,\lambda_{D^*} 0}^{\lambda_\psi} \, \mathcal{A}_{D^*,00}^{\lambda_{D^*}}$$

where:
- expression $\mathcal{A}_{B,\lambda_\psi 0}^{\lambda_B}$ is the amplitude for $B \to \psi K$ with $\psi$ helicity $\lambda_\psi$ and $K$ helicity 0 (parent $B$ has helicity $\lambda_B = 0$)
- expression $\mathcal{A}_{\psi,\lambda_{D^*} 0}^{\lambda_\psi}$ is the amplitude for $\psi \to D^* \bar{D}$ with $D^*$ helicity $\lambda_{D^*}$, $\bar{D}$ helicity 0, and parent $\psi$ helicity $\lambda_\psi$
- expression $\mathcal{A}_{D^*,00}^{\lambda_{D^*}}$ is the amplitude for $D^* \to D^0 \pi$ with both daughters having helicity 0 and parent $D^*$ helicity $\lambda_{D^*}$

## Spin Configurations

The spin assignments are:
- the $B$: $J^P = 0^-$ (pseudoscalar)
- the $\psi$: $J^P = 1^-$ (vector)
- the $D^*$: $J^P = 1^-$ (vector)
- the $K, \bar{D}, D^0, \pi$: $J^P = 0^-$ (pseudoscalars)

For each two-body decay:
- decay: $B \to \psi K$: $0^- \to 1^- + 0^-$ (requires $l=1, s=1$)
- decay: $\psi \to D^* \bar{D}$: $1^- \to 1^- + 0^-$ (requires $l=1, s=1$)
- decay: $D^* \to D^0 \pi$: $1^- \to 0^- + 0^-$ (requires $l=1, s=0$)

## Implementation Details

The implementation uses:
- **TwoBodySystem**: Stores masses and spin quantum numbers
- **TwoBodyDecay**: Combines a TwoBodySystem with a VertexFunction
- **VertexFunction**: Contains the recoupling coefficient and form factor
- **SimpleCascade**: Chains three TwoBodyDecay objects together
- **SphericalAngles**: Stores $(\cos\theta, \phi)$ for each decay vertex

The amplitude computation follows the cascade structure, computing each two-body decay in sequence and summing over intermediate helicity states.

"""

# ╔═╡ 38d52789-aac6-4db3-831a-90c39881660e
# get angles in a decay
program = (
    MeasureInvariant(:m1sq, (1,)),
    MeasureInvariant(:m2sq, (2,)),
    MeasureInvariant(:m3sq, (3,)),
    MeasureInvariant(:m4sq, (4,)),
    MeasureInvariant(:mBsq, (1, 2, 3, 4)),
    MeasureInvariant(:mψsq, (1, 2, 3)),
    MeasureInvariant(:mDKsq, (3, 4)),
    MeasureInvariant(:mDxsq, (1, 2)),
    # 
    # ToHelicityFrame((1, 2, 3, 4)),
    MeasureCosThetaPhi(:vars_B, (1, 2, 3)),
    # 
    ToHelicityFrame((1, 2, 3)),
    MeasureCosThetaPhi(:vars_ψ, (1, 2)),
    # 
    ToHelicityFrame((1, 2)),
    MeasureCosThetaPhi(:vars_Dx, (1))
);

# ╔═╡ 6d4ace44-4e8a-4fff-a969-a907ce47175b
md"""
### Three-Body Decays
"""

# ╔═╡ 8a77defd-a7cf-431a-b66f-6fdb4c28f3f7
md"""
## Implementation 3b
"""

# ╔═╡ f77b192b-23b4-4f74-87c9-b3d927b0d8d6
begin
    struct DalitzAndDecay{T}
        σs::MandelstamTuple{T}
        cosθ::T
        ϕ::T
    end

    function ThreeBodyDecays.amplitude(model::Union{ThreeBodyDecay,DecayChain}, dd::DalitzAndDecay; refζs)
        (; σs, cosθ, ϕ) = dd
        jDx = 1
        _O = amplitude(model, σs; refζs) # order: -1,0,1
        _Dh = [
            wignerD(jDx, λ, 0, ϕ, cosθ, 0.0)
            for λ in -1:1] .|> conj # order: -1,0,1
        total_amp = sum(reshape(_O, 3) .* _Dh)
        return total_amp
    end
end

# ╔═╡ 2e2238d9-5155-488e-befc-28df85fad85b
md"""
## Implementation of 2b
"""

# ╔═╡ 250d34e5-3f9c-428b-91aa-ec0d85206b54
## Below is an implementation of cascade decay dynamics
begin
    @kwarg struct TwoBodyMasses
        m1::Float64
        m2::Float64
        m0::Float64
    end
    TwoBodyMasses(m1, m2; m0) = TwoBodyMasses(; m1, m2, m0)
    # 
    @kwarg struct TwoBodySpins
        two_h1::Int
        two_h2::Int
        two_h0::Int
    end
    function TwoBodySpins(v1, v2;
        two_h0=nothing, h0=nothing)
        !(h0 === nothing) && return TwoBodySpins(;
            two_h1=x2(v1), two_h2=x2(v2), two_h0=x2(h0))
        TwoBodySpins(; two_h1, two_h2, two_h0)
    end
    function ThreeBodyDecays.amplitude(
        ch::RecouplingLS, two_λs::TwoBodySpins, two_js::TwoBodySpins)
        return amplitude(ch,
            (two_λs.two_h1, two_λs.two_h2),
            (two_js.two_h0, two_js.two_h1, two_js.two_h2))
    end
    # 
    struct TwoBodySystem
        ms::TwoBodyMasses
        two_js::TwoBodySpins
    end
    struct TwoBodyDecay{VF<:VertexFunction}
        tbs::TwoBodySystem
        vf::VF
    end
    # 
    @kwarg struct SphericalAngles
        cosθ::Float64
        ϕ::Float64
    end
    SphericalAngles(nt::NamedTuple) = SphericalAngles(; nt.cosθ, nt.ϕ)
    function ThreeBodyDecays.amplitude(
        ch::TwoBodyDecay, angles::SphericalAngles, two_λs::TwoBodySpins;
        verbose=false)

        verbose && println("=====\nangles:", angles, "\ntwo_λs:", two_λs)
        (; vf, tbs) = ch
        (; ms, two_js) = tbs
        _recoupling = amplitude(vf.h, two_λs, two_js)
        verbose && @show _recoupling
        (; m0, m1, m2) = tbs.ms
        _ff = vf.ff(m0^2, m1^2, m2^2)
        p = HadronicLineshapes.breakup(m0, m1, m2)
        verbose && @show _ff
        verbose && @show p
        two_j0 = two_js.two_h0
        two_λ0 = two_λs.two_h0
        two_Δλ = two_λs.two_h1 - two_λs.two_h2
        (; ϕ, cosθ) = angles
        D_conj = wignerD_doublearg(two_j0, two_λ0, two_Δλ, ϕ, cosθ, 0) |> conj
        verbose && @show D_conj
        return D_conj * _recoupling * _ff
    end
    # 
    struct SimpleCascade{C1,C2,C3}
        ch1::C1
        ch2::C2
        ch3::C3
    end
    function ThreeBodyDecays.amplitude(c::SimpleCascade, (Ω1, Ω2, Ω3); verbose=false)
        return sum(
            amplitude(c.ch1, Ω1, TwoBodySpins(λψ, 0; h0=0); verbose) *
            amplitude(c.ch2, Ω2, TwoBodySpins(λDx, 0; h0=λψ); verbose) *
            amplitude(c.ch3, Ω3, TwoBodySpins(0, 0; h0=λDx); verbose)
            for λψ in -1:1, λDx in -1:1)
    end
    # 
    FourVectors.FourVector(p::JSON.Object) =
        FourVector(p["px"], p["py"], p["pz"]; E=p["E"])
end

# ╔═╡ bc772bda-5498-4fe4-8c67-d6bb330be5b5
# get four-vectors from json
objs = let
    json_path = joinpath(@__DIR__, "..", "..", "..", "archive", "data", "crosscheck_event.json")
    event = JSON.parsefile(json_path)
    four_vectors_json = event["four_vectors"]
    # 
    pD = FourVector(four_vectors_json["D"])
    pD0 = FourVector(four_vectors_json["D0"])
    pK = FourVector(four_vectors_json["K"])
    pπ = FourVector(four_vectors_json["pi"])
    # 
    (pD0, pπ, pD, pK)
end;

# ╔═╡ 5875e393-3faf-48ed-9e65-160805f993ab
_, results = apply_decay_instruction(program, objs);

# ╔═╡ 90cd53d8-221f-486b-9379-6534775f7b2f
results

# ╔═╡ c57f9d9b-d0c4-4b69-80f9-24a59bb7b6bb
ch_3b = let
    mA = results.m1sq |> sqrt
    mB = results.m1sq |> sqrt
    m1 = results.mDxsq |> sqrt
    m2 = results.m3sq |> sqrt
    m3 = results.m4sq |> sqrt
    m0 = results.mBsq |> sqrt
    tbs = ThreeBodySystem(
        ThreeBodyMasses(m1, m2, m3; m0),
        ThreeBodySpins(1, 0, 0; h0=0)
    )
    d = 3.0
    DecayChain(;
        k=3,
        two_j=2,
        Xlineshape=x -> 1,
        HRk=VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(d)),
        Hij=VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(d)),
        tbs
    )
end

# ╔═╡ c8048bdf-ee1d-4b24-8c5a-20e776f43490
cascade_B_ψK_DxD_Dπ = let
    mB = results.mBsq |> sqrt
    mψ = results.mψsq |> sqrt
    mDx = results.mDxsq |> sqrt
    mK = results.m4sq |> sqrt
    mD = results.m3sq |> sqrt
    mπ = results.m2sq |> sqrt
    mD0 = results.m1sq |> sqrt
    # 
    tbs_B = TwoBodySystem(TwoBodyMasses(mψ, mK; m0=mB), TwoBodySpins(1, 0; h0=0))
    tbs_ψ = TwoBodySystem(TwoBodyMasses(mDx, mD; m0=mψ), TwoBodySpins(1, 0; h0=1))
    tbs_Dx = TwoBodySystem(TwoBodyMasses(mD0, mπ; m0=mDx), TwoBodySpins(0, 0; h0=1))
    #
    d0 = 3 # 1/GeV
    BW_FF(l, s, d) = VertexFunction(RecouplingLS((2l, 2s)), BlattWeisskopf{l}(d))
    # 
    ch_B = TwoBodyDecay(tbs_B, BW_FF(1, 1, d0))
    ch_ψ = TwoBodyDecay(tbs_ψ, BW_FF(1, 1, d0))
    ch_Dx = TwoBodyDecay(tbs_Dx, VertexFunction(RecouplingLS((2, 0))))
    # 
    SimpleCascade(ch_B, ch_ψ, ch_Dx)
end

# ╔═╡ 11b1b12f-b653-4763-b6d8-ca1a60e996ca
Ωs = (SphericalAngles(results.vars_B),
    SphericalAngles(results.vars_ψ),
    SphericalAngles(results.vars_Dx));

# ╔═╡ 13feffb0-b4dd-439b-83c0-763b319b6f20
dalitz_dpd = let
    σs_dpd = Invariants(ch_3b.tbs.ms; σ3=results.mψsq, σ1=results.mDKsq)
    DalitzAndDecay(
        σs_dpd,
        Ωs[3].cosθ,
        Ωs[3].ϕ,
    )
end

# ╔═╡ 651a5159-90cc-470a-a057-74d36abaa177
begin
    D_hdh_D = amplitude(ch_3b, dalitz_dpd; refζs=(1, 1, 1, 1))
    Dh_Dh_D = amplitude(cascade_B_ψK_DxD_Dπ, Ωs) * sqrt(3)
    Dh_Dh_D, D_hdh_D
end

# ╔═╡ Cell order:
# ╟─f15f6b55-11a4-4b95-9c65-dd324925592e
# ╠═db4f869a-fac9-11f0-313e-7b298f5edaf2
# ╠═bc772bda-5498-4fe4-8c67-d6bb330be5b5
# ╠═38d52789-aac6-4db3-831a-90c39881660e
# ╠═5875e393-3faf-48ed-9e65-160805f993ab
# ╠═90cd53d8-221f-486b-9379-6534775f7b2f
# ╠═c8048bdf-ee1d-4b24-8c5a-20e776f43490
# ╠═11b1b12f-b653-4763-b6d8-ca1a60e996ca
# ╟─6d4ace44-4e8a-4fff-a969-a907ce47175b
# ╠═c57f9d9b-d0c4-4b69-80f9-24a59bb7b6bb
# ╠═651a5159-90cc-470a-a057-74d36abaa177
# ╠═13feffb0-b4dd-439b-83c0-763b319b6f20
# ╟─8a77defd-a7cf-431a-b66f-6fdb4c28f3f7
# ╠═f77b192b-23b4-4f74-87c9-b3d927b0d8d6
# ╟─2e2238d9-5155-488e-befc-28df85fad85b
# ╠═250d34e5-3f9c-428b-91aa-ec0d85206b54
