using Arrow
using B2DxDK
using CascadeDecays
using CascadeDecaysIO
using DataFrames
using JSON

println("=================================================================")
println(" B2DxDK Amplitude Model Serialization with CascadeDecaysIO.jl")
println("=================================================================")

# 1. Kinematics specification
masses = SystemMasses(
    B2DxDK.nominal_mass["D0"],
    B2DxDK.nominal_mass["pi"],
    B2DxDK.nominal_mass["D"],
    B2DxDK.nominal_mass["K"];
    m0 = B2DxDK.nominal_mass["Bp"],
)
particle_labels = ("D0", "pi+", "D-", "K+", "B+")

# 2. Build full 19-chain model
cascade = build_all_resonance_cascade()
println("Built full model with $(length(cascade.chains)) chains across $(length(all_resonance_names)) resonances.")

# 3. Load crosscheck data for validation
crosscheck_data_path = joinpath(B2DxDK.data_dir, "crosscheck.arrow")
df = DataFrame(Arrow.Table(crosscheck_data_path))
n_events = nrow(df)
println("Loaded $n_events cross-check events from $crosscheck_data_path.")

# Compute checksums on first 5 events
checksums = []
param_points = []
for i in 1:min(5, n_events)
    pt = event_point(df[i, :])
    val = only(amplitude(cascade, pt))
    push!(checksums, Dict(
        "distribution" => "Bplus_to_D0_pi_D_K_unpolarized_intensity",
        "point" => "crosscheck_event_$i",
        "value" => abs2(val),
        "amplitude_real" => real(val),
        "amplitude_imag" => imag(val),
    ))
    push!(param_points, Dict(
        "name" => "crosscheck_event_$i",
        "parameters" => [
            Dict("name" => "weight", "value" => df.weight[i]),
            Dict("name" => "D0_E", "value" => df.D0_E[i]),
            Dict("name" => "pip_E", "value" => df.pip_E[i]),
            Dict("name" => "Dm_E", "value" => df.Dm_E[i]),
            Dict("name" => "Kp_E", "value" => df.Kp_E[i]),
        ],
    ))
end

# 4. Create document
doc = amplitudeSerializationDict(
    cascade,
    masses;
    particle_labels,
    name = "Bplus_to_D0_pi_D_K_unpolarized_intensity",
    variables = ["m1_2sq", "m1_2_3sq", "m3_4sq"],
    domains = [
        Dict(
            "name" => "phase_space",
            "type" => "product_domain",
            "axes" => [
                Dict("name" => "m1_2sq", "min" => (1.86483 + 0.13957)^2, "max" => (5.27934 - 1.86965 - 0.493677)^2),
                Dict("name" => "m1_2_3sq", "min" => (2.01026 + 1.86965)^2, "max" => (5.27934 - 0.493677)^2),
                Dict("name" => "m3_4sq", "min" => (1.86965 + 0.493677)^2, "max" => (5.27934 - 2.01026)^2),
            ],
        ),
    ],
    misc = Dict(
        "analysis" => "B2DxDK",
        "decay" => "B+ -> D- D*+ K+",
        "reference_inspire" => "2794793",
        "generator" => "CascadeDecaysIO.jl",
        "amplitude_model_checksums" => checksums,
    ),
    parameter_points = param_points,
)

# 5. Write full model JSON
out_file = joinpath(B2DxDK.data_dir, "b2dxdk_amplitude_model.json")
writeJson(out_file, doc)
println("\n[1/3] Successfully serialized full amplitude model to:")
println("      $out_file ($(filesize(out_file)) bytes)")

# 6. Write per-resonance JSONs
resonances_dir = joinpath(B2DxDK.data_dir, "resonances")
mkpath(resonances_dir)
println("\n[2/3] Serializing per-resonance models to $resonances_dir:")
for res_name in all_resonance_names
    res_cascade = build_resonance_cascade(res_name)
    safe_name = replace(res_name, "/" => "_", "(" => "_", ")" => "_", "." => "p")
    res_path = joinpath(resonances_dir, "$(safe_name).json")
    writeJson(
        res_path,
        res_cascade,
        masses;
        particle_labels,
        name = "resonance_$(safe_name)",
        variables = ["m1_2sq", "m1_2_3sq", "m3_4sq"],
        misc = Dict(
            "resonance" => res_name,
            "chains" => resonance_chain_names(res_name),
            "analysis" => "B2DxDK",
        ),
    )
    println("      - $(res_name) ($(length(res_cascade.chains)) chains) -> $(basename(res_path))")
end

# 7. Verification: Read back full model and check all 100 events
println("\n[3/3] Verifying readJson round-trip across all $n_events cross-check events...")
read_cascade, read_system, read_doc = readJson(out_file)

max_delta = 0.0
for i in 1:n_events
    pt = event_point(df[i, :])
    orig_amp = only(amplitude(cascade, pt))
    read_amp = only(amplitude(read_cascade, pt))
    delta = abs(orig_amp - read_amp)
    global max_delta = max(max_delta, delta)
    if delta > 1e-10
        error("Discrepancy in event $i: delta = $delta")
    end
end

println("      All $n_events events verified: max amplitude difference = $(max_delta)")
println("\n=================================================================")
println(" Amplitude serialization and round-trip verification SUCCESSFUL!")
println("=================================================================")
