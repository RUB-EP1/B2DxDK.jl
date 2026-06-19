using Arrow
using DataFrames
using Printf

const repo_root = normpath(joinpath(@__DIR__, ".."))
const data_path = joinpath(repo_root, "data", "crosscheck_4b.arrow")
const expected_seed = 4_040_404
const expected_events = 1_000

function max_stored_mismatch(df::DataFrame, names)
    max_abs = 0.0
    for name in names
        delta_re = df[!, Symbol(name * "_julia_re")] .- df[!, Symbol(name * "_tfpwa_re")]
        delta_im = df[!, Symbol(name * "_julia_im")] .- df[!, Symbol(name * "_tfpwa_im")]
        max_abs = max(max_abs, maximum(hypot.(delta_re, delta_im)))
    end
    total_delta = hypot.(
        df.total_julia_re .- df.total_tfpwa_re,
        df.total_julia_im .- df.total_tfpwa_im,
    )
    max_abs = max(max_abs, maximum(total_delta))
    return max_abs, total_delta
end

include(joinpath(@__DIR__, "..", "notebooks", "all_resonances_sampled_comparison.jl"))

println("All-resonance 4b amplitude crosscheck")
println("=====================================")
println("Input: ", data_path)
println()

isfile(data_path) || error("Missing crosscheck file: ", data_path)
df = DataFrame(Arrow.Table(data_path))
n_eval = nrow(df)
n_eval == expected_events || error("Expected $(expected_events) events, got $(n_eval)")
all(df.seed .== expected_seed) || error("Unexpected seed in crosscheck_4b.arrow")

println("Loaded events: ", n_eval)
println(@sprintf("  fixed seed = %d", expected_seed))
println("Recomputing Julia amplitudes from stored four-vectors...")

recomputed_total = ComplexF64[]
for idx in 1:n_eval
    sampled_p4 = Dict(
        "D0" => [df.D0_E[idx], df.D0_px[idx], df.D0_py[idx], df.D0_pz[idx]],
        "pi" => [df.pip_E[idx], df.pip_px[idx], df.pip_py[idx], df.pip_pz[idx]],
        "D" => [df.Dm_E[idx], df.Dm_px[idx], df.Dm_py[idx], df.Dm_pz[idx]],
        "K" => [df.Kp_E[idx], df.Kp_px[idx], df.Kp_py[idx], df.Kp_pz[idx]],
    )
    ctx = event_context(sampled_p4)
    push!(recomputed_total, sum(selected_cd_amplitude(ctx, name) for name in all_resonance_names))
end

recomputed_delta = [abs(recomputed_total[i] - (df.total_julia_re[i] + im * df.total_julia_im[i])) for i in 1:n_eval]
max_recompute = maximum(recomputed_delta)

stored_max, stored_total_delta = max_stored_mismatch(df, all_resonance_names)

println()
println("Stored TF-PWA vs stored Julia:")
println(@sprintf("  max |delta|      = %.6e", stored_max))
println(@sprintf("  mean total |delta| = %.6e", mean(stored_total_delta)))
println()
println("Recomputed Julia vs stored Julia total:")
println(@sprintf("  max |delta|      = %.6e", max_recompute))

stored_max <= 1e-9 || error("Stored amplitudes differ beyond tolerance")
max_recompute <= 1e-12 || error("Recomputed Julia amplitudes differ from stored values")
println("PASS: crosscheck_4b.arrow is self-consistent.")
println()
println("Done. Evaluated ", n_eval, " flat-4b events.")
