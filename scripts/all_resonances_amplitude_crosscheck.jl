using Arrow
using DataFrames
using Printf

include(joinpath(@__DIR__, "all_resonances_model.jl"))

const data_path = joinpath(data_dir, "crosscheck.arrow")
const amplitude_reference_path = joinpath(data_dir, "crosscheck_amplitudes_reference.txt")

# =============================================================================
# Evaluate on crosscheck.arrow (100-event subset of b-decay-events.arrow)
# =============================================================================

function load_amplitude_reference(path::String)
    lines = collect(eachline(path))
    length(lines) >= 2 || error("Missing amplitude reference rows: ", path)
    header = split(lines[1], '\t')
    rows = Vector{Vector{Float64}}()
    for line in lines[2:end]
        fields = split(line, '\t')
        length(fields) == length(header) || continue
        push!(rows, parse.(Float64, fields))
    end
    return header, rows
end

function amplitude_row(idx, df, component_amps)
    total = sum(component_amps)
    return Float64[
        idx,
        df.weight[idx],
        reduce(vcat, [[real(amp), imag(amp)] for amp in component_amps])...,
        real(total),
        imag(total),
    ]
end

function max_amplitude_delta(reference_path::String, df, resonance_amps_by_event)
    ref_header, ref_rows = load_amplitude_reference(reference_path)
    length(ref_rows) == length(resonance_amps_by_event) ||
        error("Reference row count ($(length(ref_rows))) must match events ($(length(resonance_amps_by_event)))")

    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for (idx, component_amps) in pairs(resonance_amps_by_event)
        ref_row = ref_rows[idx]
        computed = amplitude_row(idx, df, component_amps)
        length(ref_row) == length(computed) || error("Row width mismatch at event ", idx)
        for (a, b) in zip(ref_row, computed)
            delta = abs(a - b)
            max_abs_delta = max(max_abs_delta, delta)
            denom = max(abs(a), abs(b), 1.0)
            max_rel_delta = max(max_rel_delta, delta / denom)
        end
    end
    return max_abs_delta, max_rel_delta, ref_header
end

println("All-resonance amplitude crosscheck on crosscheck.arrow")
println("=====================================================")
println("Input:  ", data_path)
println("Model input rows (one per chain): ", nrow(resonance_chains_df))
println()

df = DataFrame(Arrow.Table(data_path))
n_eval = nrow(df)
println("Loaded events: ", n_eval)
println(@sprintf("  sum(weight) = %.6e", sum(df.weight)))
println("Building resonance cascade...")

cascade = build_all_resonance_cascade()
println("Computing per-resonance amplitudes...")

resonance_amps_by_event = Vector{Vector{ComplexF64}}()
sizehint!(resonance_amps_by_event, n_eval)

for idx in 1:n_eval
    point = event_point(df[idx, :])
    push!(
        resonance_amps_by_event,
        ComplexF64[
            only(amplitude(cascade[resonance_chain_names(name)], point))
            for name in all_resonance_names
        ],
    )
end

println()
println("Per-event total amplitude |A| (first 5 events):")
for idx in 1:min(5, n_eval)
    total = sum(resonance_amps_by_event[idx])
    println(@sprintf("  event %3d  weight=%.6e  |A|=%.6e", idx, df.weight[idx], abs(total)))
end

isfile(amplitude_reference_path) || error("Missing amplitude reference: ", amplitude_reference_path)
max_abs_delta, max_rel_delta, _ = max_amplitude_delta(amplitude_reference_path, df, resonance_amps_by_event)

println()
println("Comparison to reference (", basename(amplitude_reference_path), "):")
println(@sprintf("  max |delta|      = %.6e", max_abs_delta))
println(@sprintf("  max rel |delta| = %.6e", max_rel_delta))
max_abs_delta <= 1e-10 || error("Amplitudes do not match reference within tolerance")
println("PASS: amplitudes reproduced.")
println()
println("Done. Evaluated ", n_eval, " events.")
