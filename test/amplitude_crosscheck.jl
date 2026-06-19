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

function compute_resonance_amplitudes(df, cascade)
    n_eval = nrow(df)
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
    return resonance_amps_by_event
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
