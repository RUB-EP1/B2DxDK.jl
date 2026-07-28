# One-off validation: particle-2 algebra vs RecouplingLS / CascadeDecays / B2DxDK.
# Run: julia --project=. test/particle2_algebra.jl
using B2DxDK
using CascadeDecays
using DataFrames
using Arrow
using LinearAlgebra
using Printf
using ThreeBodyDecays

const λs = (-2, 0, 2)
const ROOT_LS = ((0, 0), (2, 2), (4, 4))

η(two_λ2) = isodd((2 - two_λ2) ÷ 2) ? -1 : 1

h_code(two_ls, two_l) = ThreeBodyDecays.amplitude(RecouplingLS(two_ls), (two_l, two_l), (0, 2, 2))

function H_cascade(two_ls, two_l)
    v = Vertex(RecouplingLS(two_ls), NoFormFactor())
    return CascadeDecays.routed_vertex_amplitude(v, (1.0, 1.0, 1.0), (0, two_l, two_l), (0, 2, 2), (cosθ = 1.0, ϕ = 0.0))
end

H_tfpwa_rec(two_ls, two_l) = ThreeBodyDecays.amplitude(B2DxDK.MissingParticleTwoPhaseLS(two_ls), (two_l, two_l), (0, 2, 2))

analytic_cg(L, lam) = if L == 0
    (-1)^(1 - lam) / sqrt(3)
elseif L == 1
    lam == 0 ? 0.0 : (lam > 0 ? -1 / sqrt(2) : 1 / sqrt(2))
else
    lam == 0 ? sqrt(2 / 3) : 1 / sqrt(6)
end

function run_validation()
    println("Particle-2 algebra vs code")
    println("=" ^ 60)

    max1 = maximum(
        abs(H_cascade(two_ls, two_l) - η(two_l) * h_code(two_ls, two_l))
        for two_ls in ROOT_LS for two_l in λs
    )
    @printf "TEST 1  CascadeDecays: H = η(λ₂)·h   max err = %.3e  %s\n" max1 (max1 < 1e-12 ? "PASS" : "FAIL")

    max2 = maximum(
        abs(η(two_l) * H_tfpwa_rec(two_ls, two_l) - h_code(two_ls, two_l))
        for two_ls in ROOT_LS for two_l in λs
    )
    @printf "TEST 2  TFPWA rec: η·(η·h) = h       max err = %.3e  %s\n" max2 (max2 < 1e-12 ? "PASS" : "FAIL")

    C = [h_code(two_ls, two_l) for two_l in λs, two_ls in ROOT_LS]
    P = diagm([1.0, -1.0, 1.0])
    g = ones(3)
    h_sum = C * g
    H_sum = [sum(H_cascade(two_ls, two_l) for two_ls in ROOT_LS) for two_l in λs]
    @printf "TEST 3  h = C·g                     max err = %.3e  %s\n" maximum(abs.(h_sum - C * g)) "PASS"
    @printf "TEST 3  H = P·C·g                   max err = %.3e  %s\n" maximum(abs.(H_sum - P * C * g)) (maximum(abs.(H_sum - P * C * g)) < 1e-12 ? "PASS" : "FAIL")

    max4 = maximum(
        abs(h_code(((0, 0), (2, 2), (4, 4))[j], two_l) - analytic_cg(j - 1, two_l ÷ 2))
        for two_l in λs for j in 1:3
    )
    @printf "TEST 4  analytic CG coefficients     max err = %.3e  %s\n" max4 (max4 < 1e-10 ? "PASS" : "FAIL")

    g = [1.2, -0.7, 0.5]
    println("TEST 5  summed vertex (wrong = h, correct = P·C·g):")
    for two_l in λs
        h_tot = sum(g[j] * h_code(two_ls, two_l) for (j, two_ls) in enumerate(ROOT_LS))
        H_tot = sum(g[j] * H_cascade(two_ls, two_l) for (j, two_ls) in enumerate(ROOT_LS))
        @printf "         λ=%+d  wrong=% .6f  correct=% .6f  Δ=% .6f\n" (two_l ÷ 2) h_tot H_tot (H_tot - h_tot)
    end

    g_test = [1.2, -0.7, 0.5]
    err7 = maximum(abs.(inv(C) * (C * g_test) - g_test))
    @printf "TEST 7  LS inversion C⁻¹(Cg)=g      max err = %.3e  %s\n" err7 (err7 < 1e-12 ? "PASS" : "FAIL")

    maxL1 = maximum(abs(H_cascade((2, 2), two_l) - h_code((2, 2), two_l)) for two_l in λs)
    @printf "TEST 9  L=1 only: H = h             max err = %.3e  %s\n" maxL1 (maxL1 < 1e-12 ? "PASS" : "FAIL")

    row_l0 = first([r for r in eachrow(B2DxDK.resonance_chain_rows("X1(2900)")) if B2DxDK.vertex_l(r.root_two_ls) == 0])
    row_l1 = first([r for r in eachrow(B2DxDK.resonance_chain_rows("X1(2900)")) if B2DxDK.vertex_l(r.root_two_ls) == 1])
    @printf "TEST 10 table recoupling: L0=%s L1=%s (expected :missing_particle2, :standard)\n" row_l0.root_recoupling row_l1.root_recoupling

    df = DataFrame(Arrow.Table(joinpath(B2DxDK.data_dir, "crosscheck.arrow")))
    max_cross = 0.0
    ref_lines = collect(eachline(joinpath(B2DxDK.data_dir, "crosscheck_amplitudes_reference.txt")))
    ref_rows = [parse.(Float64, split(l, '\t')) for l in ref_lines[2:end]]
    cascade = B2DxDK.build_all_resonance_cascade()
    for idx in 1:nrow(df)
        pt = B2DxDK.event_point(df[idx, :])
        amps = ComplexF64[
            sum(only(CascadeDecays.amplitude(cascade[name], pt)) for name in B2DxDK.resonance_chain_names(n))
            for n in B2DxDK.all_resonance_names
        ]
        total = sum(amps)
        ref = ref_rows[idx]
        computed = vcat([idx, df.weight[idx]], reduce(vcat, [[real(a), imag(a)] for a in amps]), [real(total), imag(total)])
        for (a, b) in zip(computed, ref)
            max_cross = max(max_cross, abs(a - b))
        end
    end
    @printf "TEST 11 crosscheck vs reference      max err = %.3e  %s\n" max_cross (max_cross < 1e-10 ? "PASS" : "FAIL")

    println("=" ^ 60)
    println("C matrix from code:")
    show(IOContext(stdout, :compact => true), C)
    println()
end

run_validation()
