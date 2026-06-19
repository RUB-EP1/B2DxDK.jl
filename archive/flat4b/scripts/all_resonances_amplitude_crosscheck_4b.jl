using Arrow
using DataFrames
using Printf
using Statistics

const flat4b_root = normpath(joinpath(@__DIR__, ".."))
const data_path = joinpath(flat4b_root, "data", "crosscheck_4b.arrow")
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

include(joinpath(@__DIR__, "all_resonances_model_4b.jl"))

function run_crosscheck()
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

    max_recompute = 0.0
    max_recompute_component = 0.0
    worst = (0, "", 0.0)

    for idx in 1:n_eval
        ctx = event_context_from_row(df[idx, :])
        total = 0.0 + 0.0im
        for name in all_resonance_names
            amp = resonance_amplitude(ctx, name)
            ref = df[idx, Symbol(name * "_julia_re")] + im * df[idx, Symbol(name * "_julia_im")]
            delta = abs(amp - ref)
            if delta > max_recompute_component
                max_recompute_component = delta
                worst = (idx, name, delta)
            end
            total += amp
        end
        max_recompute = max(max_recompute, abs(total - (df.total_julia_re[idx] + im * df.total_julia_im[idx])))
    end

    stored_max, stored_total_delta = max_stored_mismatch(df, all_resonance_names)

    println()
    println("Stored TF-PWA vs stored Julia:")
    println(@sprintf("  max |delta|      = %.6e", stored_max))
    println(@sprintf("  mean total |delta| = %.6e", mean(stored_total_delta)))
    println()
    println("Recomputed model vs stored Julia:")
    println(@sprintf("  max total |delta|      = %.6e", max_recompute))
    println(@sprintf("  max component |delta|  = %.6e", max_recompute_component))
    println("  worst component: event ", worst[1], " ", worst[2])

    stored_max <= 1e-9 || error("Stored amplitudes differ beyond tolerance")
    max_recompute <= 1e-12 || error("Recomputed Julia amplitudes differ from stored values")
    max_recompute_component <= 1e-9 || error("Recomputed model differs from stored per-resonance Julia amplitudes")
    println("PASS: crosscheck_4b.arrow is self-consistent.")
    println()
    println("Done. Evaluated ", n_eval, " flat-4b events.")
end

run_crosscheck()
