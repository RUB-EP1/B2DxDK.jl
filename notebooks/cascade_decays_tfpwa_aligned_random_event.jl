# CascadeDecays aligned to TF-PWA-sampled Psi(4040) events
#
# This script extends the single-event random-event comparison by:
# 1. sampling many TF-PWA phase-space events for Bp -> D K D0 pi,
# 2. evaluating the isolated TF-PWA Psi(4040) amplitude on those events,
# 3. evaluating the aligned Julia CascadeDecays amplitude on the same events,
# 4. saving event-by-event results to a text file, and
# 5. creating a Dalitz-style heatmap of mean Re(A_TF-PWA - A_CD).

using CascadeDecays
using DelimitedFiles
using FourVectors
using HadronicLineshapes
using JSON
using Plots
using Printf
using Statistics
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

const psi_bw_form_factor = BlattWeisskopf{1}(3.0)
const psi_bw_q0 = breakup(nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])
const psi_bw_gsq =
    nominal_mass["Psi(4040)"] * psi_width / (2psi_bw_q0) *
    nominal_mass["Psi(4040)"] / psi_bw_form_factor(psi_bw_q0)^2

function build_package_native_chain(topology, psi_lineshape)
    vertices = (
        (((1, 2), 3), 4) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        ((1, 2), 3) => VertexFunction(RecouplingLS((2, 2)), BlattWeisskopf{1}(3.0)),
        (1, 2) => VertexFunction(RecouplingLS((2, 0))),
    )
    propagators = (
        (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0.0im)),
        ((1, 2), 3) => (
            two_j = 2,
            lineshape = psi_lineshape,
        ),
    )
    return DecayChain(topology; propagators, vertices)
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

repo_root, analysis_dir, output_path, n_events = sys.argv[1:5]
n_events = int(n_events)

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
sample_phase_space_list = [np.asarray(v) for v in generator.generate(n_events)]
sampled_p4 = dict(zip(["D", "K", "D0", "pi"], [v.tolist() for v in sample_phase_space_list]))

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
tfpwa_amp = dg.get_amp(phsp_variables).numpy().reshape(-1)

payload = {
    "seed": seed,
    "p4": sampled_p4,
    "tfpwa_amplitude_re": np.real(tfpwa_amp).tolist(),
    "tfpwa_amplitude_im": np.imag(tfpwa_amp).tolist(),
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
"""
    write(path, script)
end

function sample_events_and_reference(n_events::Int)
    repo_root = normpath(joinpath(@__DIR__, ".."))
    analysis_dir = joinpath(repo_root, "Analysis")
    python_exe = resolve_tfpwa_python()

    output_data = mktemp() do output_path, io
        close(io)
        mktemp() do script_path, script_io
            close(script_io)
            write_sampling_script(script_path)
            cmd = `$(python_exe) $(script_path) $(repo_root) $(analysis_dir) $(output_path) $(n_events)`
            run(cmd)
            return JSON.parsefile(output_path)
        end
    end

    p4_data = output_data["p4"]
    n_loaded = length(p4_data["D"])
    sampled_events = Vector{Dict{String, Vector{Float64}}}(undef, n_loaded)
    for idx in 1:n_loaded
        event = Dict{String, Vector{Float64}}()
        for name in ["D", "K", "D0", "pi"]
            event[name] = Float64.(p4_data[name][idx])
        end
        sampled_events[idx] = event
    end

    tfpwa_amp = ComplexF64.(Float64.(output_data["tfpwa_amplitude_re"]), Float64.(output_data["tfpwa_amplitude_im"]))
    return Int(output_data["seed"]), sampled_events, tfpwa_amp
end

function sample_event_and_reference()
    seed, sampled_events, tfpwa_amp = sample_events_and_reference(1)
    return seed, sampled_events[1], tfpwa_amp[1]
end

function fourvector_from_tfpwa(v::AbstractVector{<:Real})
    return FourVector(v[2], v[3], v[4]; E = v[1])
end

function theta_beta(angle_struct)
    return acos(values(angle_struct)[1])
end

function phi_alpha(angle_struct)
    return values(angle_struct)[2]
end

function event_to_fourvectors(sampled_p4::Dict{String, Vector{Float64}})
    pD0 = fourvector_from_tfpwa(sampled_p4["D0"])
    piplus = fourvector_from_tfpwa(sampled_p4["pi"])
    pDminus = fourvector_from_tfpwa(sampled_p4["D"])
    pKplus = fourvector_from_tfpwa(sampled_p4["K"])
    return pD0, piplus, pDminus, pKplus
end

function compute_cd_observables(sampled_p4::Dict{String, Vector{Float64}}, topology, total_factor)
    pD0, piplus, pDminus, pKplus = event_to_fourvectors(sampled_p4)
    objs = (pD0, piplus, pDminus, pKplus)
    P_Dx = pD0 + piplus
    P_psi = P_Dx + pDminus
    P_B = P_psi + pKplus

    system = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., mass(P_B)^2))
    x = cascade_kinematics(topology, system, objs)
    psi_bw_event = MultichannelBreitWigner(
        nominal_mass["Psi(4040)"],
        [(; gsq = psi_bw_gsq, ma = mass(P_Dx), mb = nominal_mass["D"], l = 1, d = 3.0)],
    )
    package_chain_event = build_package_native_chain(topology, psi_bw_event)
    a_package = amplitude(package_chain_event, system, x, (0, 0, 0, 0, 0))
    a_corrected = a_package * total_factor

    event_mass = Dict(
        "Bp" => mass(P_B),
        "Psi(4040)" => mass(P_psi),
        "Dst" => mass(P_Dx),
        "D" => mass(pDminus),
        "K" => mass(pKplus),
        "D0" => mass(pD0),
        "pi" => mass(piplus),
    )

    return (
        pD0 = pD0,
        piplus = piplus,
        pDminus = pDminus,
        pKplus = pKplus,
        P_Dx = P_Dx,
        P_psi = P_psi,
        P_B = P_B,
        system = system,
        x = x,
        event_mass = event_mass,
        a_package = a_package,
        a_corrected = a_corrected,
        psi_factor_package = psi_bw_event(event_mass["Psi(4040)"]^2),
        root_angles = vertex_angles(topology, x, (((1, 2), 3), 4)),
        psi_angles = vertex_angles(topology, x, ((1, 2), 3)),
        dst_angles = vertex_angles(topology, x, (1, 2)),
    )
end

function save_batch_results_txt(path::String, rows)
    header = [
        "event",
        "m2_D0pi",
        "m2_D0piD",
        "tfpwa_re",
        "tfpwa_im",
        "cd_raw_re",
        "cd_raw_im",
        "cd_corr_re",
        "cd_corr_im",
        "rel_delta_re",
        "rel_delta_im",
        "D_E", "D_px", "D_py", "D_pz",
        "K_E", "K_px", "K_py", "K_pz",
        "D0_E", "D0_px", "D0_py", "D0_pz",
        "pi_E", "pi_px", "pi_py", "pi_pz",
    ]

    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            vals = Any[
                row.event,
                row.m2_dst,
                row.m2_psi,
                real(row.tfpwa_amp),
                imag(row.tfpwa_amp),
                real(row.cd_raw_amp),
                imag(row.cd_raw_amp),
                real(row.cd_corr_amp),
                imag(row.cd_corr_amp),
                row.rel_delta_re,
                row.rel_delta_im,
                row.sampled_p4["D"]...,
                row.sampled_p4["K"]...,
                row.sampled_p4["D0"]...,
                row.sampled_p4["pi"]...,
            ]
            println(io, join(string.(vals), '\t'))
        end
    end
end

function dalitz_heatmap(path::String, rows; nbins::Int = 60, component::Symbol = :re)
    xs = [row.m2_dst for row in rows]
    ys = [row.m2_psi for row in rows]
    zs = component === :re ? [row.rel_delta_re for row in rows] : [row.rel_delta_im for row in rows]

    xedges = collect(range(minimum(xs), maximum(xs); length = nbins + 1))
    yedges = collect(range(minimum(ys), maximum(ys); length = nbins + 1))
    sums = zeros(Float64, nbins, nbins)
    counts = zeros(Int, nbins, nbins)

    for (x, y, z) in zip(xs, ys, zs)
        ix = clamp(searchsortedlast(xedges, x), 1, nbins)
        iy = clamp(searchsortedlast(yedges, y), 1, nbins)
        if !isnan(z)
            sums[ix, iy] += z
            counts[ix, iy] += 1
        end
    end

    mean_grid = fill(NaN, nbins, nbins)
    for ix in 1:nbins, iy in 1:nbins
        if counts[ix, iy] > 0
            mean_grid[ix, iy] = sums[ix, iy] / counts[ix, iy]
        end
    end

    display_grid = similar(mean_grid)
    for ix in 1:nbins, iy in 1:nbins
        v = mean_grid[ix, iy]
        display_grid[ix, iy] = isnan(v) ? NaN : sign(v) * log10(1 + abs(v))
    end

    xcenters = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    ycenters = (yedges[1:end-1] .+ yedges[2:end]) ./ 2

    plt = heatmap(
        xcenters,
        ycenters,
        display_grid',
        xlabel = "m^2(D0, pi) [GeV^2]",
        ylabel = "m^2(D0, pi, D) [GeV^2]",
        title = component === :re ? "Signed-log mean eventwise relative Delta Re" : "Signed-log mean eventwise relative Delta Im",
        colorbar_title = component === :re ? "(Re(amp_CD) - Re(amp_TF-PWA))/Re(amp_TF-PWA)" : "(Im(amp_CD) - Im(amp_TF-PWA))/Im(amp_TF-PWA)",
        aspect_ratio = :auto,
        c = :balance,
        background_color = :white,
        background_color_inside = :white,
        colorbar_tickfontsize = 8,
        right_margin = 18Plots.mm,
    )
    savefig(plt, path)
end

function framework_comparison_plot(path::String, rows; nbins::Int = 60, component::Symbol = :re)
    xs = [row.m2_dst for row in rows]
    ys = [row.m2_psi for row in rows]
    tf_vals = [abs2(row.tfpwa_amp) for row in rows]
    cd_vals = [abs2(row.cd_corr_amp) for row in rows]

    xedges = collect(range(minimum(xs), maximum(xs); length = nbins + 1))
    yedges = collect(range(minimum(ys), maximum(ys); length = nbins + 1))
    tf_sums = zeros(Float64, nbins, nbins)
    cd_sums = zeros(Float64, nbins, nbins)
    counts = zeros(Int, nbins, nbins)

    for (x, y, tfv, cdv) in zip(xs, ys, tf_vals, cd_vals)
        ix = clamp(searchsortedlast(xedges, x), 1, nbins)
        iy = clamp(searchsortedlast(yedges, y), 1, nbins)
        tf_sums[ix, iy] += tfv
        cd_sums[ix, iy] += cdv
        counts[ix, iy] += 1
    end

    tf_grid = fill(NaN, nbins, nbins)
    cd_grid = fill(NaN, nbins, nbins)
    for ix in 1:nbins, iy in 1:nbins
        if counts[ix, iy] > 0
            tf_grid[ix, iy] = tf_sums[ix, iy] / counts[ix, iy]
            cd_grid[ix, iy] = cd_sums[ix, iy] / counts[ix, iy]
        end
    end

    log_display(v) = isnan(v) ? NaN : log10(1 + v)
    tf_display = log_display.(tf_grid)
    cd_display = log_display.(cd_grid)

    xcenters = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    ycenters = (yedges[1:end-1] .+ yedges[2:end]) ./ 2
    zmax = maximum(abs, vcat(vec(tf_display[.!isnan.(tf_display)]), vec(cd_display[.!isnan.(cd_display)])))

    left_title = "TF-PWA mean |A|^2"
    right_title = "CascadeDecays mean |A|^2"
    cbar_title = "log10(1 + <|A|^2>)"

    p1 = heatmap(
        xcenters,
        ycenters,
        tf_display',
        xlabel = "m^2(D0, pi) [GeV^2]",
        ylabel = "m^2(D0, pi, D) [GeV^2]",
        title = left_title,
        colorbar_title = cbar_title,
        aspect_ratio = :auto,
        c = :viridis,
        clim = (0, zmax),
        background_color = :white,
        background_color_inside = :white,
        left_margin = 12Plots.mm,
        bottom_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )
    p2 = heatmap(
        xcenters,
        ycenters,
        cd_display',
        xlabel = "m^2(D0, pi) [GeV^2]",
        ylabel = "m^2(D0, pi, D) [GeV^2]",
        title = right_title,
        colorbar_title = cbar_title,
        aspect_ratio = :auto,
        c = :viridis,
        clim = (0, zmax),
        background_color = :white,
        background_color_inside = :white,
        left_margin = 12Plots.mm,
        bottom_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )
    plt = plot(p1, p2; layout = (1, 2), size = (1900, 750), margin = 6Plots.mm)
    savefig(plt, path)
end

function batch_scan_and_plot(topology, total_factor; n_events::Int = 5000)
    println()
    println("Batch scan: sample many TF-PWA events and compare to the Julia CascadeDecays amplitude.")
    println("  Requested number of sampled events: ", n_events)

    seed, sampled_events, tfpwa_amps = sample_events_and_reference(n_events)
    println("  TF random seed for batch sampling: ", seed)
    println("  Received ", length(sampled_events), " sampled events from TF-PWA.")

    rows = NamedTuple[]
    sizehint!(rows, length(sampled_events))
    for (idx, sampled_p4) in enumerate(sampled_events)
        obs = compute_cd_observables(sampled_p4, topology, total_factor)
        delta = tfpwa_amps[idx] - obs.a_corrected
        tf_re = real(tfpwa_amps[idx])
        tf_im = imag(tfpwa_amps[idx])
        rel_delta_re = abs(tf_re) > 1e-12 ? (real(obs.a_corrected) - tf_re) / tf_re : NaN
        rel_delta_im = abs(tf_im) > 1e-12 ? (imag(obs.a_corrected) - tf_im) / tf_im : NaN
        push!(rows, (
            event = idx,
            sampled_p4 = sampled_p4,
            m2_dst = obs.event_mass["Dst"]^2,
            m2_psi = obs.event_mass["Psi(4040)"]^2,
            tfpwa_amp = tfpwa_amps[idx],
            cd_raw_amp = obs.a_package,
            cd_corr_amp = obs.a_corrected,
            delta = delta,
            rel_delta_re = rel_delta_re,
            rel_delta_im = rel_delta_im,
        ))
    end

    repo_root = normpath(joinpath(@__DIR__, ".."))
    txt_path = joinpath(repo_root, "notebooks", "cascade_decays_tfpwa_random_event_scan.txt")
    png_path_re = joinpath(repo_root, "notebooks", "cascade_decays_tfpwa_random_event_delta_heatmap.png")
    png_path_im = joinpath(repo_root, "notebooks", "cascade_decays_tfpwa_random_event_delta_heatmap_im.png")
    compare_png_path_re = joinpath(repo_root, "notebooks", "cascade_decays_tfpwa_framework_comparison_re.png")

    save_batch_results_txt(txt_path, rows)
    dalitz_heatmap(png_path_re, rows; component = :re)
    dalitz_heatmap(png_path_im, rows; component = :im)
    framework_comparison_plot(compare_png_path_re, rows; component = :re)

    println("  Saved event-by-event results to:")
    println("    ", txt_path)
    println("  Saved Dalitz-style heatmaps to:")
    println("    ", png_path_re)
    println("    ", png_path_im)
    println("  Saved framework comparison plot to:")
    println("    ", compare_png_path_re)
    rel_re_vals = [row.rel_delta_re for row in rows if !isnan(row.rel_delta_re)]
    rel_im_vals = [row.rel_delta_im for row in rows if !isnan(row.rel_delta_im)]
    println(@sprintf("  Mean relative Delta Re over all events = %.6e", mean(rel_re_vals)))
    println(@sprintf("  Mean relative Delta Im over all events = %.6e", mean(rel_im_vals)))
    println(@sprintf("  Mean |Delta A| over all events         = %.6e", mean(map(row -> abs(row.delta), rows))))
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

    topology = DecayTopology((((1, 2), 3), 4))
    fb = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["Psi(4040)"], nominal_mass["K"])
    fpsi = mismatch_factor(1, 3.0, nominal_mass["Psi(4040)"], nominal_mass["Dst"], nominal_mass["D"])
    total_factor = fb * fpsi

    obs = compute_cd_observables(sampled_p4, topology, total_factor)
    pD0 = obs.pD0
    piplus = obs.piplus
    pDminus = obs.pDminus
    pKplus = obs.pKplus
    event_mass = obs.event_mass
    a_package = obs.a_package
    a_corrected = obs.a_corrected
    psi_factor_package = obs.psi_factor_package
    root_angles = obs.root_angles
    psi_angles = obs.psi_angles
    dst_angles = obs.dst_angles

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
                     theta_beta(root_angles), phi_alpha(root_angles)))
    println(@sprintf("  Psi -> Dst D:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     theta_beta(psi_angles), phi_alpha(psi_angles)))
    println(@sprintf("  Dst -> D0 pi:        theta(beta)= % .12f, phi(alpha)= % .12f",
                     theta_beta(dst_angles), phi_alpha(dst_angles)))
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
    println("  Propagator: MultichannelBreitWigner with event m(D*) for q and nominal m(D*) for q0.")
    println("  Psi(4040) factor at sigma = m(Psi)^2")
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

    batch_scan_and_plot(topology, total_factor; n_events = 50000)
end

main()
