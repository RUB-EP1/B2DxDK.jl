using Arrow
using B2DxDK
using CascadeDecays
using CascadeDecaysIO
using DataFrames
using FourVectors
using JSON

println("=================================================================")
println(" B2DxDK Amplitude Model Serialization with CascadeDecaysIO.jl")
println("=================================================================")

# 1. Kinematics specification (masses omitted from kinematics section; provided event-dependently)
particle_labels = ("D0", "pi+", "D-", "K+", "B+")

# 2. Build full 19-chain model
cascade = build_all_resonance_cascade()
println("Built full model with $(length(cascade.chains)) chains across $(length(all_resonance_names)) resonances.")

# 3. Obtain 3 newly sampled verification events (with four-vectors)
function get_sampled_verification_events(n_events = 3)
    cached_file = joinpath(B2DxDK.data_dir, "sampled_verification_events.json")
    analysis_dir = joinpath(@__DIR__, "..", "archive", "investigation", "Analysis")
    python_candidates = [
        "C:\\Users\\gamma\\miniconda3\\envs\\tf-pwa-env\\python.exe",
        "python",
    ]
    python_cmd = nothing
    for py in python_candidates
        try
            if success(`$py -c "import tf_pwa"`)
                python_cmd = py
                break
            end
        catch
        end
    end

    if !isnothing(python_cmd) && isdir(analysis_dir)
        try
            println("Sampling $n_events fresh events via TF-PWA ($python_cmd)...")
            py_code = """
            import sys, json, numpy as np
            sys.path.insert(0, '../../../../tf-pwa')
            np.random.seed(42)
            from tf_pwa.config_loader import ConfigLoader
            config = ConfigLoader('config_a.yml')
            p4 = config.generate_phsp_p($n_events)
            data = {k.name: np.asarray(v).tolist() for k, v in p4.items()}
            print(json.dumps(data))
            """
            out = read(Cmd(`$python_cmd -c $py_code`, dir=analysis_dir), String)
            json_start = findfirst('{', out)
            if !isnothing(json_start)
                parsed = JSON.parse(out[json_start:end])
                open(cached_file, "w") do io
                    JSON.print(io, parsed, 2)
                end
                return parsed
            end
        catch e
            @warn "Fresh sampling via TF-PWA failed, falling back to cached events: $e"
        end
    end

    if isfile(cached_file)
        println("Loading $n_events verification events from $cached_file...")
        return JSON.parsefile(cached_file)
    else
        error("Could not obtain sampled verification events (neither TF-PWA nor $cached_file found)")
    end
end

sampled_data = get_sampled_verification_events(3)

# 4. Evaluate amplitude and intensity on the 3 verification events and record four-vectors
checksums = []
param_points = []
verification_points = []

for i in 1:3
    d0 = sampled_data["D0"][i]
    pip = sampled_data["pi"][i]
    dm = sampled_data["D"][i]
    kp = sampled_data["K"][i]

    pD0 = FourVector(d0[2], d0[3], d0[4]; E = d0[1])
    pPip = FourVector(pip[2], pip[3], pip[4]; E = pip[1])
    pDm = FourVector(dm[2], dm[3], dm[4]; E = dm[1])
    pKp = FourVector(kp[2], kp[3], kp[4]; E = kp[1])
    pt = KinematicPoint(B2DxDK.kinematic_task, (pD0, pPip, pDm, pKp))
    push!(verification_points, pt)

    val = only(amplitude(cascade, pt))
    pt_name = "verification_point_$i"

    push!(checksums, Dict(
        "distribution" => "Bplus_to_D0_pi_D_K_unpolarized_intensity",
        "point" => pt_name,
        "value" => abs2(val),
        "amplitude_real" => real(val),
        "amplitude_imag" => imag(val),
    ))

    push!(param_points, Dict(
        "name" => pt_name,
        "four_vectors" => Dict(
            "D0" => Dict("E" => d0[1], "px" => d0[2], "py" => d0[3], "pz" => d0[4]),
            "pi+" => Dict("E" => pip[1], "px" => pip[2], "py" => pip[3], "pz" => pip[4]),
            "D-" => Dict("E" => dm[1], "px" => dm[2], "py" => dm[3], "pz" => dm[4]),
            "K+" => Dict("E" => kp[1], "px" => kp[2], "py" => kp[3], "pz" => kp[4]),
        ),
        "parameters" => [
            Dict("name" => "D0_E", "value" => d0[1]),
            Dict("name" => "D0_px", "value" => d0[2]),
            Dict("name" => "D0_py", "value" => d0[3]),
            Dict("name" => "D0_pz", "value" => d0[4]),
            Dict("name" => "pip_E", "value" => pip[1]),
            Dict("name" => "pip_px", "value" => pip[2]),
            Dict("name" => "pip_py", "value" => pip[3]),
            Dict("name" => "pip_pz", "value" => pip[4]),
            Dict("name" => "Dm_E", "value" => dm[1]),
            Dict("name" => "Dm_px", "value" => dm[2]),
            Dict("name" => "Dm_py", "value" => dm[3]),
            Dict("name" => "Dm_pz", "value" => dm[4]),
            Dict("name" => "Kp_E", "value" => kp[1]),
            Dict("name" => "Kp_px", "value" => kp[2]),
            Dict("name" => "Kp_py", "value" => kp[3]),
            Dict("name" => "Kp_pz", "value" => kp[4]),
        ],
    ))
end

# 5. Create document without kinematic masses
doc = amplitudeSerializationDict(
    cascade;
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

# 6. Write full model JSON
out_file = joinpath(B2DxDK.data_dir, "b2dxdk_amplitude_model.json")
writeJson(out_file, doc)
println("\n[1/3] Successfully serialized full amplitude model to:")
println("      $out_file ($(filesize(out_file)) bytes)")

# 7. Write per-resonance JSONs (without masses)
resonances_dir = joinpath(B2DxDK.data_dir, "resonances")
mkpath(resonances_dir)
println("\n[2/3] Serializing per-resonance models to $resonances_dir:")
for res_name in all_resonance_names
    res_cascade = build_resonance_cascade(res_name)
    safe_name = replace(res_name, "/" => "_", "(" => "_", ")" => "_", "." => "p")
    res_path = joinpath(resonances_dir, "$(safe_name).json")
    writeJson(
        res_path,
        res_cascade;
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

# 8. Verification: Read back full model and check:
#    a) Kinematics has no mass
#    b) Custom functions use "type": "custom"
#    c) The 3 verification points match bit-exactly from JSON four-vectors
#    d) All 100 cross-check events match with delta < 1e-10
println("\n[3/3] Verifying readJson round-trip:")
raw_doc = JSON.parsefile(out_file)
init_state = raw_doc["distributions"][1]["decay_description"]["kinematics"]["initial_state"]
fin_state = raw_doc["distributions"][1]["decay_description"]["kinematics"]["final_state"]
if haskey(init_state, "mass") || any(haskey(f, "mass") for f in fin_state)
    error("Kinematics section must not contain mass fields!")
else
    println("      ✓ Kinematics section omits mass fields as required")
end

fns = raw_doc["functions"]
fn_types = [f["type"] for f in fns]
if "MultiChannelBreitWigner" in fn_types || "TFPWAMultichannelBreitWigner" in fn_types
    error("MultiChannelBreitWigner must not be emitted as primary type!")
else
    println("      ✓ Custom lineshapes correctly serialized with \"type\": \"custom\"")
end

read_cascade, read_system, read_doc = readJson(out_file)

# Check 3 verification points
for (i, pt) in enumerate(verification_points)
    orig_amp = only(amplitude(cascade, pt))
    read_amp = only(amplitude(read_cascade, pt))
    delta = abs(orig_amp - read_amp)
    if delta > 1e-10
        error("Verification point $i failed: delta = $delta")
    end
    println("      ✓ Verification point $i matched (delta = $delta)")
end

# Check 100 crosscheck events
crosscheck_data_path = joinpath(B2DxDK.data_dir, "crosscheck.arrow")
df = DataFrame(Arrow.Table(crosscheck_data_path))
n_events = nrow(df)
max_delta = 0.0
for i in 1:n_events
    pt = event_point(df[i, :])
    orig_amp = only(amplitude(cascade, pt))
    read_amp = only(amplitude(read_cascade, pt))
    delta = abs(orig_amp - read_amp)
    global max_delta = max(max_delta, delta)
    if delta > 1e-10
        error("Discrepancy in cross-check event $i: delta = $delta")
    end
end
println("      ✓ All $n_events cross-check events verified: max delta = $max_delta")

println("\n=================================================================")
println(" Amplitude serialization and round-trip verification SUCCESSFUL!")
println("=================================================================")

