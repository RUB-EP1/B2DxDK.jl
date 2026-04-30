# CascadeDecays aligned to a TF-PWA-sampled Psi(4040) event
#
# This script is the random-event analogue of
# notebooks/cascade_decays_tfpwa_aligned.jl. It asks the local TF-PWA checkout
# to generate one phase-space event for Bp -> D K D0 pi, then feeds those
# sampled four-vectors into the Julia CascadeDecays/HadronicLineshapes setup to
# evaluate the aligned Psi(4040) complex amplitude.

using CascadeDecays
using FourVectors
using HadronicLineshapes
using JSON
using Printf
using ThreeBodyDecays: RecouplingLS, VertexFunction

const nominal_mass = Dict(
    "Bp" => 5.27934,
    "D" => 1.86965,
    "K" => 0.493677,
    "D0" => 1.86483,
    "pi" => 0.13957039,
    "Dst" => 2.01026,
    "Psi(4040)" => 4.039,
)

const psi_width = 0.08
const final_state_order = ["D0", "pi", "D", "K"]

function build_package_native_chain()
    topology = DecayTopology((((1, 2), 3), 4))
    vertices = (
        (((1, 2), 3), 4) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        ((1, 2), 3) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        (1, 2) => VertexFunction(RecouplingLS((2, 0))),
    )
    propagators = (
        (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0.0im)),
        ((1, 2), 3) => (
            two_j = 2,
            lineshape = BreitWigner(
                nominal_mass["Psi(4040)"],
                psi_width,
                nominal_mass["Dst"],
                nominal_mass["D"],
                1,
                3.0,
            ),
        ),
    )
    return topology, DecayChain(topology; propagators, vertices)
end

function mismatch_factor(l, d, m0, m1, m2)
    ff = BlattWeisskopf{l}(d)
    return 1 / ff(m0^2, m1^2, m2^2)
end

function resolve_tfpwa_python()
    candidates = String[]
    if haskey(ENV, "TFPWA_PYTHON")
        push!(candidates, ENV["TFPWA_PYTHON"])
    end
    push!(candidates, joinpath(homedir(), "miniconda3", "envs", "tf-pwa-env", "python.exe"))
    push!(candidates, "python")

    for candidate in candidates
        if occursin('\\', candidate) || occursin('/', candidate)
            isfile(candidate) && return candidate
        else
            resolved = Sys.which(candidate)
            resolved === nothing || return resolved
        end
    end
    error("Could not find a Python executable with TensorFlow/TF-PWA. Set ENV[\"TFPWA_PYTHON\"] or activate tf-pwa-env first.")
end

function write_sampling_script(path::String)
    script = raw"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
import yaml

repo_root, analysis_dir, output_path = sys.argv[1:4]

sys.path.insert(0, os.path.join(repo_root, "tf-pwa"))
sys.path.insert(0, analysis_dir)
os.chdir(analysis_dir)

import extra_amp
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import PhaseSpaceGenerator

with open("config_a.yml", "r", encoding="utf-8") as f:
    config_yml = yaml.safe_load(f)
with open("final_params_full.json", "r", encoding="utf-8") as f:
    params_dict = json.load(f)["value"]

nominal_mass = {
    "Bp": float(config_yml["particle"]["$top"]["Bp"]["mass"]),
    "D": float(config_yml["particle"]["$finals"]["D"]["mass"]),
    "K": float(config_yml["particle"]["$finals"]["K"]["mass"]),
    "D0": float(config_yml["particle"]["$finals"]["D0"]["mass"]),
    "pi": float(config_yml["particle"]["$finals"]["pi"]["mass"]),
}

seed = int(np.random.default_rng().integers(0, 2**31 - 1))
tf.random.set_seed(seed)

generator = PhaseSpaceGenerator(
    nominal_mass["Bp"],
    [nominal_mass[name] for name in ["D", "K", "D0", "pi"]],
)
sample_phase_space_list = [np.asarray(v).tolist() for v in generator.generate(1)]
sampled_p4 = dict(zip(["D", "K", "D0", "pi"], sample_phase_space_list))

config = ConfigLoader("config_a.yml")
particles = list(config.get_decay().outs)
particle_map = {p.name: p for p in particles}

p4_tfpwa = {
    particle_map["D"]: tf.constant(sampled_p4["D"], dtype=tf.float64),
    particle_map["D0"]: tf.constant(sampled_p4["D0"], dtype=tf.float64),
    particle_map["K"]: tf.constant(sampled_p4["K"], dtype=tf.float64),
    particle_map["pi"]: tf.constant(sampled_p4["pi"], dtype=tf.float64),
}

phsp_variables = config.data.cal_angle(p4_tfpwa)
phsp_variables["c"] = np.array([-1.0])

amp_model = config.get_amplitude()
dg = amp_model.decay_group
chain = dg.chains[5]

p_unit = params_dict.copy()
for key in p_unit:
    if ("total" in key or "g_ls" in key) and (key.endswith("r") or key.endswith("i")):
        p_unit[key] = 0.0
for decay in chain.chain:
    prefix = f"{decay.core.name.replace('(1+)', '(1.)')}->{'.'.join([p.name.replace('(1+)', '(1.)') for p in decay.outs])}"
    for key in p_unit:
        if prefix in key and ("total" in key or "g_ls" in key) and key.endswith("_0r"):
            p_unit[key] = 1.0

config.set_params(p_unit)
dg.set_used_chains([5])
tfpwa_amp = dg.get_amp(phsp_variables).numpy().reshape(-1)[0]

payload = {
    "seed": seed,
    "p4": sampled_p4,
    "tfpwa_amplitude": {
        "re": float(np.real(tfpwa_amp)),
        "im": float(np.imag(tfpwa_amp)),
    },
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
"""
    write(path, script)
end

function sample_event_and_reference()
    repo_root = normpath(joinpath(@__DIR__, ".."))
    analysis_dir = joinpath(repo_root, "Analysis")
    python_exe = resolve_tfpwa_python()

    output_data = mktemp() do output_path, io
        close(io)
        mktemp() do script_path, script_io
            close(script_io)
            write_sampling_script(script_path)
            cmd = `$(python_exe) $(script_path) $(repo_root) $(analysis_dir) $(output_path)`
            run(cmd)
            return JSON.parsefile(output_path)
        end
    end

    p4_data = output_data["p4"]
    sampled_p4 = Dict{String, Vector{Float64}}()
    for name in ["D", "K", "D0", "pi"]
        sampled_p4[name] = Float64.(p4_data[name][1])
    end
    tfpwa_amp = output_data["tfpwa_amplitude"]["re"] + output_data["tfpwa_amplitude"]["im"] * im
    return Int(output_data["seed"]), sampled_p4, tfpwa_amp
end

function fourvector_from_tfpwa(v::AbstractVector{<:Real})
    return FourVector(v[2], v[3], v[4]; E = v[1])
end

function main()
    println("CascadeDecays aligned Psi(4040) amplitude on a TF-PWA-sampled event")
    println("===================================================================")
    println()

    seed, sampled_p4, tfpwa_reference = sample_event_and_reference()
    println("Sampled-event setup: generate one TF-PWA phase-space event for Bp -> D K D0 pi.")
    println("  TF random seed: ", seed)
    for name in ["D", "K", "D0", "pi"]
        println(@sprintf("  sampled %2s: [%0.16f, %0.16f, %0.16f, %0.16f]",
                         name, sampled_p4[name][1], sampled_p4[name][2], sampled_p4[name][3], sampled_p4[name][4]))
    end
    println()

    pD0 = fourvector_from_tfpwa(sampled_p4["D0"])
    piplus = fourvector_from_tfpwa(sampled_p4["pi"])
    pDminus = fourvector_from_tfpwa(sampled_p4["D"])
    pKplus = fourvector_from_tfpwa(sampled_p4["K"])

    objs = (pD0, piplus, pDminus, pKplus)
    P_Dx = pD0 + piplus
    P_psi = P_Dx + pDminus
    P_B = P_psi + pKplus

    topology, package_chain = build_package_native_chain()
    system = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., mass(P_B)^2))
    x = cascade_kinematics(topology, system, objs)

    a_package = amplitude(package_chain, system, x, (0, 0, 0, 0, 0))
    psi_bw = BreitWigner(
        nominal_mass["Psi(4040)"],
        psi_width,
        nominal_mass["Dst"],
        nominal_mass["D"],
        1,
        3.0,
    )

    event_mass = Dict(
        "Bp" => mass(P_B),
        "Psi(4040)" => mass(P_psi),
        "Dst" => mass(P_Dx),
        "D" => mass(pDminus),
        "K" => mass(pKplus),
        "D0" => mass(pD0),
        "pi" => mass(piplus),
    )

    root_angles = vertex_angles(topology, x, (((1, 2), 3), 4))
    psi_angles = vertex_angles(topology, x, ((1, 2), 3))
    dst_angles = vertex_angles(topology, x, (1, 2))

    fb = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["Psi(4040)"], nominal_mass["K"])
    fpsi = mismatch_factor(1, 3.0, nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])
    total_factor = fb * fpsi
    a_corrected = a_package * total_factor
    psi_factor_package = psi_bw(event_mass["Psi(4040)"]^2)

    println("Step 1: Reconstruct intermediate four-vectors by summing daughters.")
    println("  Final-state four-vectors are the sampled inputs passed into the Julia framework.")
    println(@sprintf("  p_D0  = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pD0[4], pD0[1], pD0[2], pD0[3]))
    println(@sprintf("  p_pi  = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", piplus[4], piplus[1], piplus[2], piplus[3]))
    println(@sprintf("  p_D   = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pDminus[4], pDminus[1], pDminus[2], pDminus[3]))
    println(@sprintf("  p_K   = (E, px, py, pz) = (%.12f, %.12f, %.12f, %.12f)", pKplus[4], pKplus[1], pKplus[2], pKplus[3]))
    println("  Intermediate four-vectors (Dst, Psi(4040), Bp) are not provided directly by the package-native interface.")
    println()

    println("Step 2: Compute invariant masses from the event kinematics.")
    println("  External/root masses are direct package inputs through CascadeSystem.")
    for name in ["D0", "pi", "D", "K", "Bp"]
        println(@sprintf("  m(%s) = %.12f GeV", name, event_mass[name]))
    end
    println("  Internal masses m(Dst) and m(Psi(4040)) are routed internally by cascade_kinematics and are not printed here as direct package inputs.")
    println()

    println("Step 3: Compute helicity angles in the package convention.")
    println("  theta(beta) is reconstructed from the package-native cos(theta) output.")
    println(@sprintf("  Bp -> Psi(4040) K:   theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(root_angles.cosθ), root_angles.ϕ))
    println(@sprintf("  Psi -> Dst D:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(psi_angles.cosθ), psi_angles.ϕ))
    println(@sprintf("  Dst -> D0 pi:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     acos(dst_angles.cosθ), dst_angles.ϕ))
    println()

    println("Step 4: Compute breakup momenta.")
    println("  q2 and q0^2 are not provided directly by the package-native interface.")
    println()

    println("Step 5: Vertex model inputs from HadronicLineshapes / ThreeBodyDecays.")
    println("  Bp -> Psi K vertex:   RecouplingLS((2,2)) with BlattWeisskopf{1}(3.0)")
    println("  Psi -> Dst D vertex:  RecouplingLS((2,2)) with BlattWeisskopf{1}(3.0)")
    println("  Dst -> D0 pi vertex:  RecouplingLS((2,0))")
    println()

    println("Step 6: Particle factor for Psi(4040).")
    println("  Propagator: BreitWigner(4.039, 0.08, 2.01026, 1.86965, 1, 3.0)")
    println("  Package-native Psi(4040) factor at sigma = m(Psi)^2")
    println("    ", psi_factor_package)
    println()

    println("Step 7: Final complex amplitude and normalization correction.")
    println("  Package-native amplitude          = ", a_package)
    println("  Mismatch factor Bp vertex         = ", fb)
    println("  Mismatch factor Psi vertex        = ", fpsi)
    println("  Total mismatch factor             = ", total_factor)
    println("  Corrected package amplitude       = ", a_corrected)
    println()

    println("Step 8: Compare the corrected Julia amplitude to live TF-PWA on the same sampled event.")
    println("  Live TF-PWA amplitude             = ", tfpwa_reference)
    println("  Corrected delta to TF-PWA         = ", a_corrected - tfpwa_reference)
    delta = a_corrected - tfpwa_reference
    if abs(delta) < 1.0e-6
        println("  PASS: sampled-event Julia amplitude matches live TF-PWA within tolerance.")
    else
        println("  FAIL: sampled-event Julia amplitude does not match live TF-PWA within tolerance.")
        println("  This indicates that the fixed-event correction is not sufficient as a universal sampled-event conversion.")
    end
end

main()
