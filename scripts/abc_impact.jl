# abc_impact.jl
# ============================================================================
# "ABC impact": quantify the effect of the three TF-PWA modelling issues
# (A, B, C) documented in docs/tfpwa_review/tfpwa_modelling_issues.qmd on the
# B2DxDK fit fractions.
#
# What this script does
# ---------------------
# It rebuilds the production CascadeDecays model of B+ -> D- D*+ K+ with one
# issue "fixed" at a time, keeping ALL fitted parameters frozen at their TF-PWA
# values, and recomputes the weighted fit fractions on the full event sample
# (data/b-decay-events.arrow). The shift in each fit fraction and in the total
# interference, relative to the TF-PWA-matching baseline, is the (naive, no-refit)
# impact estimate. It also reports the MC statistical uncertainty on the baseline
# fit fractions: each row sum N_k = sum_i w_i |A_{k,i}|^2 has uncertainty
# sqrt(n)*std(row contributions), divided by the (fixed) coherent total D.
#
#   Issue A (particle-2 phase):  X1(2900) chains, flip root_remove_particle2_phase
#                                true (TF-PWA) -> false (canonical Jacob-Wick).
#   Issue B (multichannel BW):   multichannel D*D resonances, replace the
#                                TF-PWA m/m0 running-width factor by the
#                                conventional m0/m factor (== TF-PWA fix_bug1=True).
#   Issue C (running daughter masses): not exercised here -- it is invisible in the
#                                cascade (3-body) setup, where the D* is on-shell.
#
# How to run
# ----------
# Arrow is a scripts/test-only extra (not a [deps] entry), so stack a throwaway
# env that provides it on the load path:
#
#   AENV=$(mktemp -d)
#   julia --project="$AENV" -e 'using Pkg; Pkg.add(name="Arrow", version="2.8")'
#   JULIA_LOAD_PATH="@:$AENV:@stdlib" julia --project=. scripts/abc_impact.jl

using Arrow
using DataFrames
using Printf
using Statistics
using StaticArrays
using HadronicLineshapes
using CascadeDecays
using B2DxDK
const B = B2DxDK

const data_path = joinpath(data_dir, "b-decay-events.arrow")

# ---------------------------------------------------------------------------
# Corrected multichannel Breit-Wigner (bug B "fixed" == TF-PWA fix_bug1=True).
#
# TF-PWA default (ParticleBWRLS, fix_bug1=False) running-width term:
#     m0 * Gamma(m) ~ sum_i gsq_i * m  * 2p * F_i(p)^2         (factor m/m0)
# Corrected (fix_bug1=True):
#     m0 * Gamma(m) ~ sum_i gsq_i * m0^2 * 2p/sqrt(s) * F_i^2  (factor m0/m)
# Both coincide at m = m0, so Gamma_0 at the pole is unchanged; only the
# off-resonance tail differs.
# ---------------------------------------------------------------------------
struct CorrectedMultichannelBW{N} <: HadronicLineshapes.AbstractFlexFunc
    m::Float64
    channels::SVector{N,<:NamedTuple{(:gsq, :ma, :mb, :l, :d)}}
end

function (bw::CorrectedMultichannelBW)(σ::Number)
    m0 = bw.m
    mΓ = sum(bw.channels) do ch
        FF = BlattWeisskopf{ch.l}(ch.d)
        p = breakup(sqrt(σ), ch.ma, ch.mb)
        ch.gsq * m0^2 * 2p / sqrt(σ) * FF(p)^2
    end
    HadronicLineshapes.BW(σ, m0, mΓ / m0)
end
(bw::CorrectedMultichannelBW)(σ::Real) = bw(σ + 1im * eps())

# Resonances whose lineshape is a TF-PWA multichannel BWR_LS (bug B affects them).
const MULTICHANNEL_BASES =
    Set([:bwr_ls_l0, :bwr_ls_l2, :adhoc_q0_bwr_ls_l0, :adhoc_q0_bwr_ls_l2])

is_multichannel_row(row) = B.lineshape_spec(row.lineshape).base in MULTICHANNEL_BASES

# Build the lineshape for a row, optionally applying the bug-B fix.
function chain_lineshape(row; fix_bug_B::Bool)
    ls = B.build_chain_lineshape(row)
    if fix_bug_B && is_multichannel_row(row)
        return CorrectedMultichannelBW(ls.m, ls.channels)
    end
    return ls
end

# Reproduce build_chain_from_row, but with knobs for the two fixes.
function build_chain(row; fix_bug_A::Bool, fix_bug_B::Bool)
    ls = chain_lineshape(row; fix_bug_B=fix_bug_B)
    root_l = B.vertex_l(row.root_two_ls)
    daughter_l = B.vertex_l(row.daughter_two_ls)
    if row.topology == :dk
        # fix_bug_A: drop the TF-PWA workaround => keep canonical particle-2 phase
        remove_phase = row.root_remove_particle2_phase && !fix_bug_A
        return B.build_dk_chain(
            ls, row.propagator_two_j, row.root_two_ls, row.daughter_two_ls,
            root_l, daughter_l; remove_root_particle2_phase=remove_phase,
        )
    end
    return B.build_dxd_chain(
        ls, row.propagator_two_j, row.root_two_ls, row.daughter_two_ls,
        root_l, daughter_l,
    )
end

function build_cascade(name; fix_bug_A::Bool=false, fix_bug_B::Bool=false)
    rows = collect(eachrow(B.resonance_chain_rows(name)))
    chains = Tuple(build_chain(row; fix_bug_A=fix_bug_A, fix_bug_B=fix_bug_B) for row in rows)
    couplings = Tuple(row.coupling_value * row.matching_factor for row in rows)
    names = Tuple(B.chain_name(name, row) for row in rows)
    return CascadeDecay(chains, B.standard_system, B.dxd_topology; couplings=couplings, names=names)
end

# Build a merged cascade for the whole model, with optional fixes applied.
function build_full(; fix_bug_A::Bool=false, fix_bug_B::Bool=false)
    cascades = [build_cascade(name; fix_bug_A=fix_bug_A, fix_bug_B=fix_bug_B) for name in all_resonance_names]
    return merge(cascades...)
end

function amps_by_event(df, cascade)
    n = nrow(df)
    out = Vector{Vector{ComplexF64}}(undef, n)
    for idx in 1:n
        point = event_point(df[idx, :])
        out[idx] = ComplexF64[
            only(amplitude(cascade[resonance_chain_names(name)], point))
            for name in all_resonance_names
        ]
    end
    return out
end

function fit_fractions(amps, weights)
    coherent = sum(weights[i] * abs2(sum(a)) for (i, a) in pairs(amps))
    comp = [sum(weights[i] * abs2(a[k]) for (i, a) in pairs(amps)) for k in eachindex(all_resonance_names)]
    ff = comp ./ coherent
    interference = (coherent - sum(comp)) / coherent
    return ff, interference
end

# Statistical (MC) uncertainty on each fit fraction.
#
# The fit fraction is FF_k = N_k / D, where the *row sum* N_k = sum_i y_{k,i},
# y_{k,i} = w_i |A_{k,i}|^2, is a sum over n events. Treating the per-event
# contribution as a sample mean, sigma(mean) = std(y)/sqrt(n), so the row-sum
# uncertainty is sigma(N_k) = n * sigma(mean) = sqrt(n) * std(y_{k,.}).
# To turn into a fraction we divide by the total D, whose own uncertainty we
# neglect (D treated as a fixed constant). Hence sigma(FF_k) = sigma(N_k) / D.
# For the interference row, the "row sum" is the incoherent total
# I = sum_k N_k = sum_i (sum_k y_{k,i}); FF_int = (D - I)/D and, with D fixed,
# sigma(FF_int) = sqrt(n) * std(sum_k y_{k,.}) / D.
function fit_fraction_errors(amps, weights)
    n = length(amps)
    D = sum(weights[i] * abs2(sum(a)) for (i, a) in pairs(amps))
    K = length(all_resonance_names)
    y = [weights[i] * abs2(amps[i][k]) for i in 1:n, k in 1:K]   # n x K per-event row contributions
    σN = [std(@view y[:, k]) * sqrt(n) for k in 1:K]             # sigma of each row sum N_k
    σff = σN ./ D
    incoh = vec(sum(y; dims=2))                                  # per-event incoherent total
    σ_int = std(incoh) * sqrt(n) / D
    return σff, σ_int
end

# ---------------------------------------------------------------------------
println("Loading events: ", data_path)
df = DataFrame(Arrow.Table(data_path))
weights = df.weight
println("  events = ", nrow(df), "   sum(weight) = ", @sprintf("%.4e", sum(weights)))

println("Building model variants ...")
base = build_full()
fixA = build_full(; fix_bug_A=true)
fixB = build_full(; fix_bug_B=true)
fixAB = build_full(; fix_bug_A=true, fix_bug_B=true)

println("Computing amplitudes ...")
amps_base = amps_by_event(df, base)
ff_base, int_base = fit_fractions(amps_base, weights)
σff_base, σint_base = fit_fraction_errors(amps_base, weights)
ff_A, int_A = fit_fractions(amps_by_event(df, fixA), weights)
ff_B, int_B = fit_fractions(amps_by_event(df, fixB), weights)
ff_AB, int_AB = fit_fractions(amps_by_event(df, fixAB), weights)

# Baseline fit fractions with statistical (MC) uncertainty.
println()
println("Baseline (TF-PWA-matching) fit fractions with statistical uncertainty:")
@printf("  %-16s %16s\n", "component", "fit_frac_% +- stat")
for (k, name) in enumerate(all_resonance_names)
    @printf("  %-16s %10.4f +- %.4f\n", name, 100ff_base[k], 100σff_base[k])
end
@printf("  %-16s %10.4f +- %.4f\n", "interference", 100int_base, 100σint_base)

println()
println("Impact of each fix on fit fractions (percentage points, fixed - baseline):")
@printf("  %-16s %10s %12s %12s %12s\n", "component", "base_%", "dFF_A(pp)", "dFF_B(pp)", "dFF_AB(pp)")
for (k, name) in enumerate(all_resonance_names)
    @printf("  %-16s %10.4f %12.4f %12.4f %12.4f\n",
        name, 100ff_base[k], 100(ff_A[k]-ff_base[k]), 100(ff_B[k]-ff_base[k]), 100(ff_AB[k]-ff_base[k]))
end
@printf("  %-16s %10.4f %12.4f %12.4f %12.4f\n",
    "interference", 100int_base, 100(int_A-int_base), 100(int_B-int_base), 100(int_AB-int_base))

# Aggregate magnitude of change.
l1(ffx) = sum(abs.(ffx .- ff_base)) + abs((1 - sum(ffx)) - (1 - sum(ff_base)))
println()
@printf("Sum |dFF| over components (pp):  A = %.4f   B = %.4f   AB = %.4f\n",
    100sum(abs.(ff_A .- ff_base)), 100sum(abs.(ff_B .- ff_base)), 100sum(abs.(ff_AB .- ff_base)))
@printf("Max |dFF| single component (pp): A = %.4f   B = %.4f   AB = %.4f\n",
    100maximum(abs.(ff_A .- ff_base)), 100maximum(abs.(ff_B .- ff_base)), 100maximum(abs.(ff_AB .- ff_base)))
