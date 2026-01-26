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
	])
	# 
	using JSON
	using Plots
	using FourVectors
	using ThreeBodyDecays
	using HadronicLineshapes
	using PartialWaveFunctions
	using InstructionalDecayTrees
end

# ╔═╡ 96a328c0-9ac0-40c2-a219-203b1c476789
md"""
# Cascade decay of B

This notebook implements Two-Body Decay unit and a cascade decays.
The amplitude is computed for (((1,2),3),4) 
- 123,4: $B \to \psi K$ with
- 12,3: $\psi \to D^*\bar{D}$ and
- 1,2: $D^* \to D^0 π$
"""

# ╔═╡ 38d52789-aac6-4db3-831a-90c39881660e
# get angles in a decay
program = (
	MeasureInvariant(:m1sq, (1,)),
	MeasureInvariant(:m2sq, (2,)),
	MeasureInvariant(:m3sq, (3,)),
	MeasureInvariant(:m4sq, (4,)),
	MeasureInvariant(:mBsq, (1,2,3,4)),
	MeasureInvariant(:mψsq, (1,2,3)),
	MeasureInvariant(:mDxsq, (1,2)),
	# 
    ToHelicityFrame((1, 2, 3, 4)),
	MeasureCosThetaPhi(:vars_B, (1,2,3)),
	# 
    ToHelicityFrame((1, 2, 3)),
	MeasureCosThetaPhi(:vars_ψ, (1,2)),
	# 
    ToHelicityFrame((1, 2)),
	MeasureCosThetaPhi(:vars_Dx, (1))
);

# ╔═╡ 6dfdafb7-5da0-4b82-a701-307116a1d453
# cross check (two_l,two_s)
possible_ls(jp"1-",jp"0-"; jp=jp"0-") |> first,
possible_ls(jp"1-",jp"0-"; jp=jp"1-") |> first,
possible_ls(jp"0-",jp"0-"; jp=jp"1-") |> first

# ╔═╡ 2e2238d9-5155-488e-befc-28df85fad85b
md"""
## Implementation
"""

# ╔═╡ 250d34e5-3f9c-428b-91aa-ec0d85206b54
## Below is an implementation of cascade decay dynamics
begin
	@kwarg struct TwoBodyMasses
		m1::Float64
		m2::Float64
		m0::Float64
	end
	TwoBodyMasses(m1,m2; m0) = TwoBodyMasses(; m1,m2, m0)
	# 
	@kwarg struct TwoBodySpins
		two_h1::Int
		two_h2::Int
		two_h0::Int
	end
	function TwoBodySpins(v1,v2;
				 two_h0=nothing, h0=nothing)
		!(h0 === nothing) && return TwoBodySpins(;
				two_h1=x2(v1),two_h2 = x2(v2), two_h0=x2(h0))
		TwoBodySpins(; two_h1,two_h2, two_h0)
	end
	function ThreeBodyDecays.amplitude(
		ch::RecouplingLS, two_λs::TwoBodySpins, two_js::TwoBodySpins)
		return amplitude(ch,
			(two_λs.two_h1,two_λs.two_h2),
			(two_js.two_h0,two_js.two_h1,two_js.two_h2))
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
	SphericalAngles(nt::NamedTuple) = SphericalAngles(; nt.cosθ,nt.ϕ)
	function ThreeBodyDecays.amplitude(
		ch::TwoBodyDecay, angles::SphericalAngles, two_λs::TwoBodySpins)
		(; vf, tbs) = ch
		(; ms, two_js) = tbs
		_rec = amplitude(vf.h, two_λs, two_js)
		(; m0, m1, m2) = tbs.ms
		_ff = vf.ff(m0^2, m1^2, m2^2)
		two_j0 = two_js.two_h0
		two_λ0 = two_λs.two_h0
		two_Δλ = two_λs.two_h1-two_λs.two_h2
		(; ϕ, cosθ) = angles
		D_conj = wignerD_doublearg(two_j0, two_λ0, two_Δλ, ϕ, cosθ, 0) |> conj
		return sqrt(two_j0+1)*D_conj * _rec
	end
	# 
	struct SimpleCascade{C1,C2,C3}
		ch1::C1
		ch2::C2
		ch3::C3
	end
	function ThreeBodyDecays.amplitude(c::SimpleCascade, (Ω1,Ω2,Ω3))
		return sum(
			amplitude(c.ch1, Ω1, TwoBodySpins(λψ,0;h0=0)) *
			amplitude(c.ch2, Ω2, TwoBodySpins(λDx,0;h0=λψ)) *
			amplitude(c.ch3, Ω3, TwoBodySpins(0,0;h0=λDx))
		for λψ in -1:1, λDx in -1:1)
	end
	# 
	FourVectors.FourVector(p::JSON.Object) = 
		FourVector(p["px"],p["py"],p["pz"]; E=p["E"])
end

# ╔═╡ bc772bda-5498-4fe4-8c67-d6bb330be5b5
# get four-vectors from json
objs = let
	json_path = joinpath(@__DIR__, "..", "data", "crosscheck_event.json")
	event = JSON.parsefile(json_path)
	four_vectors_json = event["four_vectors"]
	# 
	pD = FourVector(four_vectors_json["D"])
	pD0 = FourVector(four_vectors_json["D0"])
	pK = FourVector(four_vectors_json["K"])
	pπ = FourVector(four_vectors_json["pi"])
	pDx = FourVector(four_vectors_json["Dx"])
	@assert pDx ≈ pD0+pπ
	# 
	(pD0, pπ, pD, pK)
end;

# ╔═╡ 5875e393-3faf-48ed-9e65-160805f993ab
_, results = apply_decay_instruction(program, objs);

# ╔═╡ 08993713-7ad1-460c-a63a-de575583aa35
results

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
	tbs_B = TwoBodySystem(TwoBodyMasses(mψ, mK; m0=mB), TwoBodySpins(1,0; h0=0))
	tbs_ψ = TwoBodySystem(TwoBodyMasses(mDx, mD; m0=mψ), TwoBodySpins(1,0; h0=1))
	tbs_Dx = TwoBodySystem(TwoBodyMasses(mD0, mπ; m0=mDx), TwoBodySpins(0,0; h0=1))
	#
	d0 = 3 # 1/GeV
	BW_FF(l,s,d) = VertexFunction(RecouplingLS((2l,2s)), BlattWeisskopf{l}(d))
	# 
	ch_B = TwoBodyDecay(tbs_B, BW_FF(1,1,d0))
	ch_ψ = TwoBodyDecay(tbs_ψ, BW_FF(1,1,d0))
	ch_Dx = TwoBodyDecay(tbs_Dx, BW_FF(1,0,d0))
	# 
	SimpleCascade(ch_B, ch_ψ, ch_Dx)
end

# ╔═╡ 11b1b12f-b653-4763-b6d8-ca1a60e996ca
Ωs = (SphericalAngles(results.vars_B),
	 SphericalAngles(results.vars_ψ),
	 SphericalAngles(results.vars_Dx));

# ╔═╡ 46065443-dc62-47a1-b65a-4287554bd4d0
amplitude(cascade_B_ψK_DxD_Dπ, Ωs)

# ╔═╡ Cell order:
# ╟─96a328c0-9ac0-40c2-a219-203b1c476789
# ╠═db4f869a-fac9-11f0-313e-7b298f5edaf2
# ╠═bc772bda-5498-4fe4-8c67-d6bb330be5b5
# ╠═38d52789-aac6-4db3-831a-90c39881660e
# ╠═5875e393-3faf-48ed-9e65-160805f993ab
# ╠═08993713-7ad1-460c-a63a-de575583aa35
# ╠═c8048bdf-ee1d-4b24-8c5a-20e776f43490
# ╠═6dfdafb7-5da0-4b82-a701-307116a1d453
# ╠═11b1b12f-b653-4763-b6d8-ca1a60e996ca
# ╠═46065443-dc62-47a1-b65a-4287554bd4d0
# ╟─2e2238d9-5155-488e-befc-28df85fad85b
# ╠═250d34e5-3f9c-428b-91aa-ec0d85206b54
