### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ fbf4886a-a2ad-4bbf-864d-9314cb0f45b1
let
	using Pkg
	Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
	Pkg.instantiate()
	# 
    using HadronicLineshapes
    using OrderedCollections
    using ThreeBodyDecays
    using LinearAlgebra
    using Measurements
    using Statistics
    using Parameters
    using DataFrames.PrettyTables
    using DataFrames
    using Setfield
    using QuadGK
    using Plots
    using Optim
    using JSON
    using Cuba
    # 
    using Random
    Random.seed!(1234)
end


# ╔═╡ 3c2b7684-59d4-4266-9365-f8b7d3a71fdd
md"""
# Implementation of $B^+\to D^-D^{*+} K^+$

- Paper link: ([inspirehep](https://inspirehep.net/literature/2794793)), ([arxiv](https://arxiv.org/pdf/2406.03156))
- Internal documsntation: ([TWiki](https://twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/Bm2DstmDpKm)), ([ANA-NOTE](https://twiki.cern.ch/twiki/pub/LHCbPhysics/Bm2DstmDpKm/Bm2DstDKm_CombFit_note_final.pdf)) 
"""

# ╔═╡ a7f75def-7006-4afe-91f9-1a73514d8b5d
theme(:boxed)

# ╔═╡ bd1c747d-a7cd-4552-a26f-824360f62831
md"""
## Building model

$B^+ \to D^- K^+ D^{*+}$
"""

# ╔═╡ db1e2d20-ded6-42e1-8ddc-e965ea1ae60e
begin
    const mB = 5.27934 # B+
    const mD = 1.86965 # D+
    const mDx = 2.01026 # Dx+: m(D) + Δm(D*,D) from PDG
    const mK = 0.493677 # K+
end;

# ╔═╡ 49e4360a-8e84-4ff6-ac9f-9610b929ca60
(two_js, pc), (_, pv) = map(["0+", "0-"]) do jp0
    ThreeBodySpinParities("0-", "0-", "1-"; jp0)
end;

# ╔═╡ f6f916c4-b18c-4dd5-9522-4b8b0f77d5e0
tbs = let
    ms = ThreeBodyMasses(mD, mK, mDx; m0 = mB)
    ThreeBodySystem(ms, two_js)
end;

# ╔═╡ c2b128a2-a057-4aef-91b3-9240cbc5471e
md"""
- Parameters of 4040 is take  from [PDG](https://pdglive.lbl.gov/Particle.action?init=0&node=M072&home=MXXX025)
- Parameters of 3920 are from [PDG](https://pdglive.lbl.gov/Particle.action?init=0&node=M050&home=MXXX025)
"""

# ╔═╡ 0f68b6d3-7b4a-4df6-915f-1962308bac8b
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

# ╔═╡ b062d8dc-4e27-4d31-b234-8757375a8071
const EFF = BreitWigner(3.85, 0.001);

# ╔═╡ 13874ee7-3c4f-47e4-a658-c0608b4c9ce7
begin
    @with_kw struct NRexp <: HadronicLineshapes.AbstractFlexFunc
        αβ::ComplexF64
        m0::Float64
    end
    (f::NRexp)(σ::Float64) = exp(f.αβ * (σ - f.m0^2))
    # 
    const ConstantLineshape = WrapFlexFunction(x -> 1.0)
end

# ╔═╡ e32aeb71-bf20-4beb-bdec-f75716bbed90
resonances =
    [
        (; jp = jp"1+", name = "EFF(1++)", lineshape = EFF),
        # 
        (; jp = jp"0-", name = "ηc(3945)", lineshape = BreitWigner(3.945, 0.13)),
        (; jp = jp"2+", name = "χc2(3930)", lineshape = BreitWigner(3.922, 0.0352)),
        (; jp = jp"1+", name = "hc(4000)", lineshape = BreitWignerSDwaves(m = 4.0, Γ = 0.184, γS = cos(0.8)^2, d = 3.0)),
        (; jp = jp"1+", name = "χc1(4010)", lineshape = BreitWignerSDwaves(m = 4.0125, Γ = 0.0627, γS = cos(0.47)^2, d = 3.0)),
        (; jp = jp"1-", name = "ψ(4040)", lineshape = BreitWigner(4.04, 0.084)),
        (; jp = jp"1+", name = "hc(4300)", lineshape = BreitWignerSDwaves(m = 4.3073, Γ = 0.058, γS = cos(1.02)^2, d = 3.0)),
        # Tcbarsbar
        (; jp = jp"0+", name = "Tcs0(2870)", lineshape = BreitWigner(m = 2.914, Γ = 0.128, ma = mD, mb = mK, l = 0, d = 3.0)),
        (; jp = jp"1-", name = "Tcs1(2900)", lineshape = BreitWigner(m = 2.887, Γ = 0.092, ma = mD, mb = mK, l = 1, d = 3.0)),
        # NR
        (; jp = jp"1-", name = "NR(1--)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0--)", lineshape = ConstantLineshape),
        (; jp = jp"1+", name = "NR(1++)", lineshape = ConstantLineshape),
        (; jp = jp"0-", name = "NR(0-+)", lineshape = NRexp(αβ = 0.11 - 0.34im, m0 = 4.35)),
    ] |> DataFrame

# ╔═╡ 8d710e27-93ec-4d81-a090-d679233d0a77
decay_chains = [
    (k = 2, resonance_name = "EFF(1++)", l = 0),
    (k = 2, resonance_name = "EFF(1++)", l = 2),
    # 
    (k = 2, resonance_name = "ηc(3945)"),
    (k = 2, resonance_name = "χc2(3930)"),
    (k = 2, resonance_name = "hc(4000)", l = 0),
    (k = 2, resonance_name = "hc(4000)", l = 2),
    (k = 2, resonance_name = "χc1(4010)", l = 0),
    (k = 2, resonance_name = "χc1(4010)", l = 2),
    (k = 2, resonance_name = "ψ(4040)"),
    (k = 2, resonance_name = "hc(4300)", l = 0),
    (k = 2, resonance_name = "hc(4300)", l = 2),
    # 
    (k = 2, resonance_name = "NR(1--)"),
    (k = 2, resonance_name = "NR(0--)"),
    (k = 2, resonance_name = "NR(1++)", l = 0),
    (k = 2, resonance_name = "NR(0-+)"),
    # 
    (k = 3, resonance_name = "Tcs0(2870)"),
    (k = 3, resonance_name = "Tcs1(2900)", L = 0),
    (k = 3, resonance_name = "Tcs1(2900)", L = 1),
    (k = 3, resonance_name = "Tcs1(2900)", L = 2),
];

# ╔═╡ 4322da22-798f-48d8-af05-216c958bab41
chains = let
    resonance_dict = LittleDict(
        resonances.name .=> NamedTuple.(eachrow(resonances)))
    # 	
    map(decay_chains) do dc
        @unpack k, resonance_name = dc
        _jp = resonance_dict[resonance_name].jp
        comprete_data = complete_l_s_L_S(_jp, tbs.two_js, [pc, pv], dc; k)
        @unpack L, S, l, s = comprete_data
        # 
        two_j = _jp.two_j
        # 
        d = 3.0
        X = resonance_dict[resonance_name].lineshape
        ff_Rk = BlattWeisskopf{div(x2(L), 2)}(d)(breakup_Rk(tbs.ms; k))
        ff_ij = BlattWeisskopf{div(x2(l), 2)}(d)(breakup_ij(tbs.ms; k))
        Xlineshape = X * ff_Rk * ff_ij
        # 
        HRk = RecouplingLS((L, S) .|> x2)
        Hij = RecouplingLS((l, s) .|> x2)
        # 
        DecayChain(; k, two_j, Xlineshape, Hij, HRk, tbs)
    end
end;

# ╔═╡ e5bbf96f-2642-40a9-a324-c5b2aea7e587
const model_pure = let
    names = getproperty.(decay_chains, :resonance_name) .*
            "_l" .* [ch.Hij.two_ls[1] |> d2 for ch in chains]
    names .*= [(ch.k == 3) ? "_L$(ch.HRk.two_ls[1] |> d2)" : "" for ch in chains]
    ThreeBodyDecay(names .=> zip(fill(1.0 + 0.0im, length(chains)), chains))
end;

# ╔═╡ 808ef024-d742-4799-8847-df7531759f4b
md"""
## Normalization
"""

# ╔═╡ 869ce97b-f2c5-4f0b-a0f1-531097bfe19d
const nMC_draft = 400_001;

# ╔═╡ 5a5601d0-45f4-49b3-aad1-faabf9cb17a7
const phsp_sample = let
    ms = masses(model_pure)
    _draft = mapslices(rand(nMC_draft, 2); dims = 2) do y
        y2σs(y, ms; k = 1)
    end |> vec
    filter(x -> isphysical(x, ms), _draft)
end;

# ╔═╡ 2648555f-3afd-4fec-99bb-11fa82ecb89d
md"""
$I_\text{tot} = c^\dagger  M  c = \sum_{i,j} c_i^* M_{i,j} c_j$

when $c = c_a + c_b$

Then,

```math
\begin{align}
I_{ab}^\text{tot} &= c_a^* M_{a,a} c_a + c_b^* M_{b,b} c_b + c_a^* M_{a,b} c_b+ + c_b^* M_{b,a} c_a\,,\\
&= M_{a,a} + M_{b,b} + (M_{a,b}+M_{b,a}),\quad\text{if } c_a=c_b=1\,,\\
&= M_{a,a} + M_{b,b} + i(M_{a,b}-M_{b,a}),\quad\text{ if } c_a=1, \, c_b=i\,.
\end{align}
```

$I_{ij} = \frac{I(c_1+c_2)-I(c_1)-I(c_2)}{2} = \Re[c_1^*X_{12} c_2]$

$I_{ij} = |c_i||c_j||X_{ij}|\cos(Δ_{ij}+\phi_j-\phi_i)$

"""

# ╔═╡ b55a1ceb-d12d-424c-80e5-9198aa83b9b2
integrals_backup_file = joinpath(@__DIR__, "..", "..", "..", "data", "backup_$(nMC_draft).json");

# ╔═╡ 5958dc67-7e6e-44ab-8c43-53cfc23a2d92
function integral_matrices(model, phsp_sample)
    precompute_amps = map(model.chains) do dc
        fs(σs) = amplitude(dc, σs)
        map(fs, phsp_sample)
    end
    A = map(Iterators.product(precompute_amps, precompute_amps)) do (_ai, _aj)
        sample1 = map(x -> sum(abs2, x), _ai + _aj)
        sample2 = map(x -> sum(abs2, x), _ai + 1im .* _aj)
        mean(sample1), mean(sample2), std(sample1), std(sample2)
    end
    m1 = getindex.(A, 1)
    m2 = getindex.(A, 2)
    δ1, δ2 = (getindex.(A, 3), getindex.(A, 4)) ./ sqrt(length(phsp_sample))
    # 
    _computation = (; m1, m2, δ1, δ2)
end

# ╔═╡ 4cabae49-ce4e-4cde-8e67-7d3c92125d75
integral_computation = if isfile(integrals_backup_file)
    _content = open(JSON.parse, integrals_backup_file)
    m1 = hcat(_content["m1"]...)
    m2 = hcat(_content["m2"]...)
    δ1 = hcat(_content["δ1"]...)
    δ2 = hcat(_content["δ2"]...)
    (; m1, m2, δ1, δ2)
else
    _computation = integral_matrices(model_pure, phsp_sample)
    open(integrals_backup_file, "w") do io
        JSON.print(io, _computation)
    end
    # 
    _computation
end;

# ╔═╡ 556560f6-7d59-4719-92a6-91b54e1b15b6
const integral_matrix = let
    m1, m2 = integral_computation
    d1 = diag(m1) / 4
    d2 = diag(m2)
    #
    sum_d = d1' .+ d1 - 2Diagonal(d1)
    # 
    m1_off_diag = m1 - Diagonal(m1)
    m1′ = Diagonal(d1) + (m1_off_diag - sum_d) / 2
    # 
    m2_off_diag = m2 - Diagonal(d2)
    m2′ = (m2_off_diag - sum_d) / 2
    #
    Hermitian(m1′ + 1im .* m2′)
end;

# ╔═╡ b41f00af-aa1e-457c-85d8-95b757b499b1
let # test if matrix is constructed correctly
    i, j = 2, 7
    # 
    c1 = [k == i for k in 1:size(integral_matrix, 1)]
    c2 = [k == j for k in 1:size(integral_matrix, 1)]
    c_rr = c1 + c2
    c_ir = c1 * 1im + c2
    # 
    @assert (c_rr' * integral_matrix * c_rr) ≈ integral_computation.m1[i, j]
    #
    @assert (c_ir' * integral_matrix * c_ir) ≈ integral_computation.m2[i, j]
    # 
    @assert integral_matrix[i, i] ≈ integral_computation.m1[i, i] / 4
    @assert integral_matrix[j, j] ≈ integral_computation.m1[j, j] / 4
end

# ╔═╡ e42021bf-a6f4-4ad3-bfa8-adbcabceffa3
md"""
## Minimization
"""

# ╔═╡ bcd8e17c-a599-4781-aad4-f73987c60d69
const nFits = 30

# ╔═╡ b613f233-0f74-4be3-a05d-9b7d771b71d5
function gloss(ϕ, X, I, δI, δtot)
    r = sqrt.(diag(I) ./ diag(X))
    _I = abs.(X) .* r .* r' .* cos.(angle.(X) .+ ϕ' .- ϕ)
    mismatch = I .- _I
    mean(abs2, mismatch ./ δI) + abs2((sum(I) - sum(_I)) / δtot)
end

# ╔═╡ 942bc39d-063a-4faf-97dc-9f7e3d361465
md"""
## Get right fractions
"""

# ╔═╡ 67f1ce27-701f-4b90-bcf9-4b61e69bea42
interference_data = let
    file_name = joinpath(@__DIR__, "..", "..", "..", "data", "interference_tf.json")
    open(JSON.parse, file_name)
end;

# ╔═╡ dc096c5c-1c0e-4356-b9dc-ea99a76c934d
function from_diag_to_matrix(representation; waves, waves_ref)
    # 
    @assert length(waves_ref) == length(waves)
    # 
    m = representation
    n = length(m)
    I = map(Iterators.product(1:n, 1:n)) do (i, j)
        _i = max(i, j)
        _j = min(i, j)
        m[_i][_j]
    end
    D = Diagonal(diag(I)) + (I - Diagonal(diag(I))) / 2
    # 
    orders = map(waves_ref) do w
        index = findfirst(x -> x == w, waves)
        index === nothing && error("Wave $w not found")
        index
    end
    # 
    D[orders, orders]
end

# ╔═╡ a01515c9-4f11-43f0-ac96-75f395c8d894
const M = from_diag_to_matrix(
    interference_data["matrix"] .* 100;
    waves = interference_data["waves"],
    waves_ref = model_pure.names);

# ╔═╡ 9fb5077c-2070-436d-974c-5a4cf9338870
const δM = from_diag_to_matrix(
    interference_data["uncertainty"] .* 100;
    waves = interference_data["waves"],
    waves_ref = model_pure.names);

# ╔═╡ 2d387906-a5e9-48b0-8967-2e68ccd1d924
const δMtot = sum(M .± δM).err

# ╔═╡ b3af8e27-8b9f-49a8-bf35-601dd67ce65e
function fit_phases(X, I; δI = fill(1.0, size(I)))
    #
    n = size(X, 1)
    function loss(ϕ, X, I, δI, δtot)
        r = sqrt.(diag(I) ./ diag(X))
        _I = abs.(X) .* r .* r' .* cos.(angle.(X) .+ ϕ' .- ϕ)
        mismatch = I .- _I
        mean(abs2, mismatch ./ δI) + abs2((sum(I) - sum(_I)) / δtot)
    end
    loss(ϕ_n1) = loss(vcat([0], ϕ_n1), X, I, δI, δMtot)
    # 
    init_unit = rand(n - 1)
    init_phases = π .* (2init_unit .- 1)
    step = 2π / 1000
    optimation_result = optimize(loss, init_phases, BFGS(
            initial_invH = x -> Matrix{eltype(x)}(step^2 * LinearAlgebra.I, length(x), length(x)),
        ), Optim.Options(iterations = 1000), autodiff = :forward)
    phases = vcat([0.0], optimation_result.minimizer)
    (; phases, optimation_result)
end

# ╔═╡ 8f7b0710-d69f-4f5c-8e93-cc0760ba7496
repeated_fits = let
    X = integral_matrix
    I = M
    r = sqrt.(diag(I) ./ diag(X))
    #
    Xbar = X .* r .* r'
    # 
    map(1:nFits) do _
        fit_phases(Xbar, I; δI = δM)
    end
end;

# ╔═╡ ca77bfef-5d44-44ed-b025-f9c6ad962d28
map(repeated_fits) do fit
    X = integral_matrix
    I = M
    r = sqrt.(diag(I) ./ diag(X))
    ϕ = fit.phases
    c = r .* cis.(fit.phases)
    # 
    c' * X * c |> real
end

# ╔═╡ 2167c35c-9ad4-4f1a-be02-945e430fe9d7
chi2_fits = map(repeated_fits) do fit
    fit.optimation_result.minimum
end;

# ╔═╡ e6097d3e-deb1-4847-8f61-6b1e6de6a646
stephist(chi2_fits, bins = range(0, 100, 100), fill = 0, alpha = 0.4, linealpha = 1)

# ╔═╡ 70654c29-85aa-4aff-af3a-4d505831c8ea
begin
    repeated_fits_sorted = sort(repeated_fits, by = x -> x.optimation_result.minimum)
    best_phases = repeated_fits_sorted[1].phases
    mismatch_interferences = repeated_fits_sorted[1].optimation_result.minimum
end

# ╔═╡ e1d3c055-f2f0-4760-ba8c-8f82445dac99
fractions = model_pure.names .=> diag(M);

# ╔═╡ 7f98134f-565d-487f-bc5a-b3ca30dd25d8
md"""
### VS Paper
Compare the files to paper tables
"""

# ╔═╡ f8e10abf-9a7d-435a-aa72-77b10ef52838
paper_matrix = let
    _name = joinpath(@__DIR__, "..", "..", "..", "data", "interference_paper.json")
    _data = open(JSON.parse, _name)
    segs = map(w -> split(w, "_")[1], model_pure.names) |> unique
    from_diag_to_matrix(
        _data["matrix"];
        waves = _data["waves"],
        waves_ref = segs)
end;

# ╔═╡ e0d11ae3-84f2-4a3c-a06d-f2eaa0389519
δpaper_matrix = let
    _name = joinpath(@__DIR__, "..", "..", "..", "data", "interference_paper.json")
    _data = open(JSON.parse, _name)
    segs = map(w -> split(w, "_")[1], model_pure.names) |> unique
    from_diag_to_matrix(
        _data["uncertainty"];
        waves = _data["waves"],
        waves_ref = segs)
end;

# ╔═╡ ac0123a3-4c0d-45bf-8c10-19513e82037e
combination_matrix = let
    segs = map(w -> split(w, "_")[1], model_pure.names)
    res = unique(segs)
    hcat([segs .== s0 for s0 in res]...)
end;

# ╔═╡ 992a634a-5864-4dbf-922d-c7ea12abccc9
partially_summed_M = let
    nR = size(combination_matrix, 2)
    map(Iterators.product(1:nR, 1:nR)) do (i, j)
        indices_i = combination_matrix[:, i]
        indices_j = combination_matrix[:, j]
        sum(M[indices_i, indices_j])
    end
end;

# ╔═╡ c4339c0d-3962-46f5-ad64-4abed0e88306
round.((partially_summed_M - paper_matrix) ./ δpaper_matrix, digits = 2)

# ╔═╡ 9fef4b8b-0152-4c5d-80e9-59d731a4b83d
md"""
### Bare couplings
and their phases
"""

# ╔═╡ 5022e509-3f16-41fc-a89a-3668cef32e03
bare_couplings = begin
    _data = open(JSON.parse, joinpath(@__DIR__, "..", "..", "..", "data", "paper_couplings.json"))["couplings"]
    Dict(k => Meta.parse(v) |> eval |> angle for (k, v) in _data)
end

# ╔═╡ 18400ff3-9a51-4bdf-aa89-f0cded804fda
bare_phases = [bare_couplings[w] for w in model_pure.names]

# ╔═╡ 68afac86-917e-46c6-9580-4cff0348d913
possible_bare_phases = let
    n = length(bare_phases)
    mapslices(rand(Bool, (20000, n)); dims = 2) do x
        flips = 2x .- 1
        phases = bare_phases .* flips
        chi2 = gloss(phases, integral_matrix, M, δM, δMtot)
        (; chi2, phases)
    end[:, 1]
end;

# ╔═╡ 06e4ad2e-033a-4454-8c10-623dcd44e96e
stephist(getproperty.(possible_bare_phases, :chi2), bins = range(0, 100, 300))

# ╔═╡ edae6b2a-48f0-49a7-be36-c5f3ac1926b8
best_bare_phases = sort(possible_bare_phases; by = x -> x.chi2)[1].phases

# ╔═╡ d115dd1c-851e-4d69-859b-6b9767764806
md"""
## Update model
"""

# ╔═╡ 15a95a02-d1d6-4e78-bdf3-6393f1f7c9a2
const model = let
    T = typeof(model_pure.couplings)
    values = model_pure.couplings .* sqrt.(diag(M) ./ diag(integral_matrix))
    phases_values = values .* cis.(best_phases) # best_phases
    @set model_pure.couplings = T(phases_values)
end

# ╔═╡ 73009655-d11f-4167-9e6c-861f61385b75
let
    n = length(repeated_fits_sorted[1].phases)
    _size = 200
    m = 3
    l = div(n, m) + 1
    plot(layout = grid(m, l), size = (_size * l, (m + 1) * _size),
        axis = false, aspect_ratio = 1, ticks = false, xlim = (-1, 1), ylim = (-1, 1))
    #
    c_abs = sqrt.(diag(M) ./ diag(integral_matrix)) |> real
    r = sqrt.(c_abs ./ maximum(c_abs))
    @assert length(r) == n
    # 
    map(1:n) do sp
        plot!(Plots.Shape(reim.(r[sp] .* cis.(range(-π, π, 100))));
            c = :green, alpha = 0.3, sp, title = model.names[sp])
    end
    map(repeated_fits_sorted[1:5]) do res
        map(enumerate(res.phases)) do (sp, p)
            plot!(r[sp] .* [0.0, cis(p)]; sp, lw = 5, xlab = "", ylab = "")
        end
    end
    map(enumerate(best_bare_phases)) do (sp, p)
        plot!(r[sp] .* [0.0, cis(p)]; sp, lw = 5, xlab = "", ylab = "",
            l = (:dash, 4, :black))
    end
    plot!()
end

# ╔═╡ 848a45c0-7802-4865-8347-00ad6cf4e573
let
    c = model.couplings
    X = integral_matrix
    r = abs.(c)
    ϕ = angle.(c)
    FF_matrix = r' .* r .* abs.(X) .* cos.(angle.(X) .+ ϕ' .- ϕ)
    # 
    clim = (-10, 10)
    c = Plots.cgrad(:terrain; rev = true)
    aspect_ratio = 1
    plot(
        heatmap(FF_matrix; aspect_ratio, clim, c, title = "julia"),
        heatmap(M; aspect_ratio, clim, c, title = "paper"),
        grid = (1, 2), size = (700, 300))
end

# ╔═╡ c0512783-68d7-4e58-b3ad-0786e9546270
gloss(best_bare_phases, integral_matrix, M, δM, sum(M .± δM).err)

# ╔═╡ 6a474b2b-b7fc-43e1-afb3-24ffd837d0f7
gloss(best_phases, integral_matrix, M, δM, sum(M .± δM).err)

# ╔═╡ 8e527f9e-a9d0-437b-8eae-e38fd052cf62
md"""
## Plotting
"""

# ╔═╡ b4626253-2914-4c7a-b39e-24fefab2673b
plot(masses(model), aspect_ratio = 1, iσx = 2, iσy = 3,
    xlab = "m²(D-Dx+) [GeV²]", ylab = "m²(D-K+) [GeV²]",
    c = palette(:viridis)) do σs
    unpolarized_intensity(model, σs)
end

# ╔═╡ 108f9da0-baa6-44ff-80e7-b864a73b428e
model.couplings' * integral_matrix * model.couplings

# ╔═╡ 4c46f63c-7c4b-451c-9283-50da3116b920
md"""
### Projections
"""

# ╔═╡ 6c56df2c-7a97-4eac-b6ff-8cd863fc9142
let
    k = 2
    ev = range(sqrt.(lims(masses(model); k))..., 90)
    # 
    plot(xlab = "m(D⁻Dˣ⁺) [GeV]")
    plot!(ev) do e
        I = Base.Fix1(unpolarized_intensity, model)
        integrand = projection_integrand(I, masses(model), e^2; k)
        e * quadgk(integrand, 0, 1)[1]
    end
    # 
    T = typeof(model.couplings)
    _model = @set model.couplings = T(model.couplings .*
                                      [occursin("χc1", n) for n in model.names])
    # 
    plot!(ev) do e
        I = Base.Fix1(unpolarized_intensity, _model)
        integrand = projection_integrand(I, masses(_model), e^2; k)
        e * quadgk(integrand, 0, 1)[1]
    end
end

# ╔═╡ 7dedd655-71a2-449d-97a2-7cc5935be854
let
    k = 3
    ev = range(sqrt.(lims(masses(model); k))..., 50)
    # \^+
    plot(xlab = "m(D⁻K⁺) [GeV]")
    plot!(ev) do e
        I = Base.Fix1(unpolarized_intensity, model)
        integrand = projection_integrand(I, masses(model), e^2; k)
        e * quadgk(integrand, 0, 1)[1]
    end
    # 
    T = typeof(model.couplings)
    # 
    _model = @set model.couplings = T(model.couplings .*
                                      [occursin("Tcs0", n) for n in model.names])
    plot!(ev) do e
        I = Base.Fix1(unpolarized_intensity, _model)
        integrand = projection_integrand(I, masses(_model), e^2; k)
        e * quadgk(integrand, 0, 1)[1]
    end
    # 
    _model = @set model.couplings = T(model.couplings .*
                                      [occursin("Tcs1", n) for n in model.names])
    plot!(ev) do e
        I = Base.Fix1(unpolarized_intensity, _model)
        integrand = projection_integrand(I, masses(_model), e^2; k)
        e * quadgk(integrand, 0, 1)[1]
    end
end

# ╔═╡ Cell order:
# ╟─3c2b7684-59d4-4266-9365-f8b7d3a71fdd
# ╠═fbf4886a-a2ad-4bbf-864d-9314cb0f45b1
# ╠═a7f75def-7006-4afe-91f9-1a73514d8b5d
# ╟─bd1c747d-a7cd-4552-a26f-824360f62831
# ╠═db1e2d20-ded6-42e1-8ddc-e965ea1ae60e
# ╠═49e4360a-8e84-4ff6-ac9f-9610b929ca60
# ╠═f6f916c4-b18c-4dd5-9522-4b8b0f77d5e0
# ╟─c2b128a2-a057-4aef-91b3-9240cbc5471e
# ╠═e32aeb71-bf20-4beb-bdec-f75716bbed90
# ╠═0f68b6d3-7b4a-4df6-915f-1962308bac8b
# ╠═b062d8dc-4e27-4d31-b234-8757375a8071
# ╠═13874ee7-3c4f-47e4-a658-c0608b4c9ce7
# ╠═8d710e27-93ec-4d81-a090-d679233d0a77
# ╠═4322da22-798f-48d8-af05-216c958bab41
# ╠═e5bbf96f-2642-40a9-a324-c5b2aea7e587
# ╟─808ef024-d742-4799-8847-df7531759f4b
# ╠═869ce97b-f2c5-4f0b-a0f1-531097bfe19d
# ╠═5a5601d0-45f4-49b3-aad1-faabf9cb17a7
# ╟─2648555f-3afd-4fec-99bb-11fa82ecb89d
# ╠═b55a1ceb-d12d-424c-80e5-9198aa83b9b2
# ╠═5958dc67-7e6e-44ab-8c43-53cfc23a2d92
# ╠═4cabae49-ce4e-4cde-8e67-7d3c92125d75
# ╠═556560f6-7d59-4719-92a6-91b54e1b15b6
# ╟─b41f00af-aa1e-457c-85d8-95b757b499b1
# ╟─e42021bf-a6f4-4ad3-bfa8-adbcabceffa3
# ╠═bcd8e17c-a599-4781-aad4-f73987c60d69
# ╠═8f7b0710-d69f-4f5c-8e93-cc0760ba7496
# ╠═ca77bfef-5d44-44ed-b025-f9c6ad962d28
# ╠═2167c35c-9ad4-4f1a-be02-945e430fe9d7
# ╠═e6097d3e-deb1-4847-8f61-6b1e6de6a646
# ╠═70654c29-85aa-4aff-af3a-4d505831c8ea
# ╠═b613f233-0f74-4be3-a05d-9b7d771b71d5
# ╠═b3af8e27-8b9f-49a8-bf35-601dd67ce65e
# ╟─942bc39d-063a-4faf-97dc-9f7e3d361465
# ╠═67f1ce27-701f-4b90-bcf9-4b61e69bea42
# ╠═dc096c5c-1c0e-4356-b9dc-ea99a76c934d
# ╠═a01515c9-4f11-43f0-ac96-75f395c8d894
# ╠═9fb5077c-2070-436d-974c-5a4cf9338870
# ╠═2d387906-a5e9-48b0-8967-2e68ccd1d924
# ╠═e1d3c055-f2f0-4760-ba8c-8f82445dac99
# ╟─7f98134f-565d-487f-bc5a-b3ca30dd25d8
# ╠═f8e10abf-9a7d-435a-aa72-77b10ef52838
# ╠═e0d11ae3-84f2-4a3c-a06d-f2eaa0389519
# ╠═ac0123a3-4c0d-45bf-8c10-19513e82037e
# ╠═992a634a-5864-4dbf-922d-c7ea12abccc9
# ╠═c4339c0d-3962-46f5-ad64-4abed0e88306
# ╟─9fef4b8b-0152-4c5d-80e9-59d731a4b83d
# ╠═5022e509-3f16-41fc-a89a-3668cef32e03
# ╠═18400ff3-9a51-4bdf-aa89-f0cded804fda
# ╠═73009655-d11f-4167-9e6c-861f61385b75
# ╠═68afac86-917e-46c6-9580-4cff0348d913
# ╠═06e4ad2e-033a-4454-8c10-623dcd44e96e
# ╠═edae6b2a-48f0-49a7-be36-c5f3ac1926b8
# ╟─d115dd1c-851e-4d69-859b-6b9767764806
# ╠═15a95a02-d1d6-4e78-bdf3-6393f1f7c9a2
# ╠═848a45c0-7802-4865-8347-00ad6cf4e573
# ╠═c0512783-68d7-4e58-b3ad-0786e9546270
# ╠═6a474b2b-b7fc-43e1-afb3-24ffd837d0f7
# ╟─8e527f9e-a9d0-437b-8eae-e38fd052cf62
# ╠═b4626253-2914-4c7a-b39e-24fefab2673b
# ╠═108f9da0-baa6-44ff-80e7-b864a73b428e
# ╟─4c46f63c-7c4b-451c-9283-50da3116b920
# ╟─6c56df2c-7a97-4eac-b6ff-8cd863fc9142
# ╟─7dedd655-71a2-449d-97a2-7cc5935be854
