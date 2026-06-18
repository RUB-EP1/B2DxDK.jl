using Arrow
using DataFrames
using Printf

include(joinpath(@__DIR__, "all_resonances_fit_fractions.jl"))

const crosscheck_data_path = joinpath(repo_root, "data", "crosscheck.arrow")
const crosscheck_reference_path = joinpath(repo_root, "data", "crosscheck_amplitudes_reference.txt")
const crosscheck_output_path = joinpath(repo_root, "data", "crosscheck_amplitudes.txt")

function save_amplitudes(path::String, df, resonance_amps_by_event)
    header = vcat(
        ["event", "weight"],
        reduce(vcat, [[name * "_re", name * "_im"] for name in all_resonance_names]),
        ["total_re", "total_im"],
    )
    open(path, "w") do io
        println(io, join(header, '\t'))
        for (idx, component_amps) in pairs(resonance_amps_by_event)
            total = sum(component_amps)
            values = Any[
                idx,
                df.weight[idx],
                reduce(vcat, [[real(amp), imag(amp)] for amp in component_amps])...,
                real(total),
                imag(total),
            ]
            println(io, join(string.(values), '\t'))
        end
    end
end

function load_amplitudes(path::String)
    isfile(path) || return nothing
    lines = collect(eachline(path))
    length(lines) >= 2 || return nothing
    header = split(lines[1], '\t')
    rows = Vector{Vector{Float64}}()
    for line in lines[2:end]
        fields = split(line, '\t')
        length(fields) == length(header) || continue
        push!(rows, parse.(Float64, fields))
    end
    return header, rows
end

function compare_amplitudes(reference_path::String, computed_path::String)
    ref_header, ref_rows = load_amplitudes(reference_path)
    cmp_header, cmp_rows = load_amplitudes(computed_path)
    ref_header == cmp_header || error("Header mismatch between reference and computed files")
    length(ref_rows) == length(cmp_rows) || error("Row count mismatch")

    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for (ref_row, cmp_row) in zip(ref_rows, cmp_rows)
        for (a, b) in zip(ref_row, cmp_row)
            delta = abs(a - b)
            max_abs_delta = max(max_abs_delta, delta)
            denom = max(abs(a), abs(b), 1.0)
            max_rel_delta = max(max_rel_delta, delta / denom)
        end
    end
    return max_abs_delta, max_rel_delta
end

function main(; compare_to = crosscheck_reference_path)
    println("All-resonance amplitudes on crosscheck.arrow")
    println("==========================================")
    println("Input:     ", crosscheck_data_path)
    println("Reference: ", compare_to)
    println("Output:    ", crosscheck_output_path)
    println("CascadeDecays version: ", pkgversion(CascadeDecays))
    println()

    isfile(crosscheck_data_path) || error("Missing crosscheck data: ", crosscheck_data_path)
    df = DataFrame(Arrow.Table(crosscheck_data_path))
    n_events = nrow(df)
    println("Loaded events: ", n_events)

    resonance_amps_by_event = Vector{Vector{ComplexF64}}()
    sizehint!(resonance_amps_by_event, n_events)

    for idx in 1:n_events
        ctx = event_context(df[idx, :])
        push!(resonance_amps_by_event, ComplexF64[selected_cd_amplitude(ctx, name) for name in all_resonance_names])
    end

    save_amplitudes(crosscheck_output_path, df, resonance_amps_by_event)
    println("Saved amplitudes to ", crosscheck_output_path)

    if compare_to !== nothing
        isfile(compare_to) || error("Missing reference amplitudes: ", compare_to)
        max_abs_delta, max_rel_delta = compare_amplitudes(compare_to, crosscheck_output_path)
        println()
        println("Comparison to reference (", basename(compare_to), "):")
        println(@sprintf("  max |delta|      = %.6e", max_abs_delta))
        println(@sprintf("  max rel |delta| = %.6e", max_rel_delta))
        max_abs_delta <= 1e-10 || error("Amplitudes do not match reference within tolerance")
        println("PASS: amplitudes reproduced.")
    end

    return resonance_amps_by_event
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_to = length(ARGS) >= 1 ? ARGS[1] : crosscheck_reference_path
    main(; compare_to = compare_to)
end
