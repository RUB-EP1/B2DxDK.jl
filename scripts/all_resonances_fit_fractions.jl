using Arrow
using B2DxDK
using CascadeDecays
using DataFrames
using Printf

const data_path = joinpath(data_dir, "b-decay-events.arrow")
const output_path = joinpath(repo_root, "scripts", "all_resonances_fit_fractions.txt")
const saved_reference_path = joinpath(repo_root, "archive", "notebooks", "all_resonances_fit_fractions.txt")

# =============================================================================
# Fit-fraction analysis
# =============================================================================

function compute_fit_fractions(component_names, component_amps_by_event, weights)
    length(weights) == length(component_amps_by_event) ||
        error("weights length ($(length(weights))) must match events ($(length(component_amps_by_event)))")
    coherent_norm = sum(weights[idx] * abs2(sum(component_amps)) for (idx, component_amps) in pairs(component_amps_by_event))
    component_norms = [
        sum(weights[idx] * abs2(component_amps[i]) for (idx, component_amps) in pairs(component_amps_by_event))
        for i in eachindex(component_names)
    ]
    fit_fractions = component_norms ./ coherent_norm
    incoherent_norm = sum(component_norms)
    interference_fraction = (coherent_norm - incoherent_norm) / coherent_norm
    return fit_fractions, interference_fraction, component_norms, coherent_norm, sum(weights)
end

function save_fit_fractions(path::String, component_names, fit_fractions, interference_fraction, component_norms)
    open(path, "w") do io
        println(io, join(["component", "sum_weight_times_abs2", "fit_fraction", "fit_fraction_percent"], '\t'))
        for (name, norm, fraction) in zip(component_names, component_norms, fit_fractions)
            println(io, join(string.((name, norm, fraction, 100fraction)), '\t'))
        end
        println(io, join(string.(("interference", "", interference_fraction, 100interference_fraction)), '\t'))
    end
end

function load_saved_fit_fractions(path::String)
    isfile(path) || return nothing
    saved = Dict{String,Float64}()
    for line in eachline(path)
        startswith(line, "component") && continue
        fields = split(line, '\t')
        length(fields) >= 3 || continue
        saved[fields[1]] = parse(Float64, fields[3])
    end
    return saved
end

println("All-resonance fit fractions from b-decay-events.arrow")
println("====================================================")
println("Input:  ", data_path)
println("Output: ", output_path)
println("Model input rows (one per chain): ", nrow(resonance_chains_df))
println()

df = DataFrame(Arrow.Table(data_path))
n_events = nrow(df)
weights = df.weight
println("Loaded events: ", n_events)
println("Using event weights from column: weight")
println(@sprintf("  sum(weight) = %.6e", sum(weights)))
println("Building resonance cascade...")

cascade = build_all_resonance_cascade()
println("Computing per-resonance amplitudes...")

resonance_amps_by_event = Vector{Vector{ComplexF64}}()
sizehint!(resonance_amps_by_event, n_events)

for idx in 1:n_events
    point = event_point(df[idx, :])
    push!(
        resonance_amps_by_event,
        ComplexF64[
            only(amplitude(cascade[resonance_chain_names(name)], point))
            for name in all_resonance_names
        ],
    )
    idx % 10_000 == 0 && println("  processed ", idx, " / ", n_events)
end

fit_fractions, interference_fraction, component_norms, coherent_norm, _ =
    compute_fit_fractions(all_resonance_names, resonance_amps_by_event, weights)
save_fit_fractions(output_path, all_resonance_names, fit_fractions, interference_fraction, component_norms)

println()
println("CascadeDecays weighted fit fractions (sum w|A|^2 / sum w|A_total|^2):")
for (name, fraction) in zip(all_resonance_names, fit_fractions)
    println(@sprintf("  %-16s %.6e  (%.4f%%)", name, fraction, 100fraction))
end
println(@sprintf("  %-16s %.6e  (%.4f%%)", "interference", interference_fraction, 100interference_fraction))
println()
println("Saved: ", output_path)

saved = load_saved_fit_fractions(saved_reference_path)
if saved === nothing
    println()
    println("No saved reference found at: ", saved_reference_path)
else
    println()
    println("Comparison to saved reference (", basename(saved_reference_path), "):")
    println(@sprintf("  %-16s %12s %12s %12s", "component", "computed", "saved", "delta"))
    max_abs_delta = 0.0
    for name in all_resonance_names
        saved_val = get(saved, name, NaN)
        idx = findfirst(==(name), all_resonance_names)
        delta = fit_fractions[idx] - saved_val
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", name, fit_fractions[idx], saved_val, delta))
    end
    if haskey(saved, "interference")
        delta = interference_fraction - saved["interference"]
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", "interference", interference_fraction, saved["interference"], delta))
    end
    println(@sprintf("Max |delta| = %.6e", max_abs_delta))
end
