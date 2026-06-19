using Arrow
using DataFrames
using JSON
using Printf
using Statistics

include(joinpath(@__DIR__, "all_resonances_model_4b.jl"))

const resonance_chain_idx = Dict(
    "X(3872)" => 0,
    "X(3915)(0-)" => 1,
    "chi(c2)(3930)" => 2,
    "X(3940)(1.)" => 3,
    "X(3993)" => 4,
    "Psi(4040)" => 5,
    "X(4300)" => 6,
    "NR(0-)SPp" => 7,
    "NR(1.)PSp" => 8,
    "NR(0-)SPm" => 9,
    "NR(1-)PPm" => 10,
    "X0(2900)" => 11,
    "X1(2900)" => 12,
)

function resolve_tfpwa_python()
    candidates = String[]
    haskey(ENV, "TFPWA_PYTHON") && push!(candidates, ENV["TFPWA_PYTHON"])
    push!(candidates, normpath(joinpath(repo_root, ".venv-tfpwa", "bin", "python")))
    push!(candidates, joinpath(homedir(), "miniconda3", "envs", "tf-pwa-env", "bin", "python"))
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
    error("Could not find a Python executable with TensorFlow and TF-PWA.")
end

const flat4b_root = normpath(joinpath(@__DIR__, ".."))
const output_path = joinpath(flat4b_root, "data", "crosscheck_4b.arrow")
const crosscheck_seed = 4_040_404
const n_crosscheck_events = 1_000

function write_crosscheck_4b_python_script(path::String)
    script = """
import json
import os
import sys

import numpy as np
import tensorflow as tf
import yaml

repo_root, analysis_dir, output_path, n_events, seed = sys.argv[1:6]
n_events = int(n_events)
seed = int(seed)
sys.path.insert(0, os.path.join(repo_root, "archive", "investigation", "tf-pwa"))
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
    "Bp": float(config_yml["particle"]["\$top"]["Bp"]["mass"]),
    "D": float(config_yml["particle"]["\$finals"]["D"]["mass"]),
    "K": float(config_yml["particle"]["\$finals"]["K"]["mass"]),
    "D0": float(config_yml["particle"]["\$finals"]["D0"]["mass"]),
    "pi": float(config_yml["particle"]["\$finals"]["pi"]["mass"]),
}

np.random.seed(seed)
tf.random.set_seed(seed)

generator = PhaseSpaceGenerator(
    nominal_mass["Bp"],
    [nominal_mass[name] for name in ["D", "K", "D0", "pi"]],
)
sample_phase_space_list = [np.asarray(v) for v in generator.generate(n_events)]
sampled_p4 = dict(zip(["D", "K", "D0", "pi"], [v.tolist() for v in sample_phase_space_list]))

config = ConfigLoader("config_a.yml")
particle_map = {p.name: p for p in list(config.get_decay().outs)}
p4_tfpwa = {
    particle_map["D"]: tf.constant(sampled_p4["D"], dtype=tf.float64),
    particle_map["D0"]: tf.constant(sampled_p4["D0"], dtype=tf.float64),
    particle_map["K"]: tf.constant(sampled_p4["K"], dtype=tf.float64),
    particle_map["pi"]: tf.constant(sampled_p4["pi"], dtype=tf.float64),
}
phsp_variables = config.data.cal_angle(p4_tfpwa)
phsp_variables["c"] = np.full(n_events, -1.0)

amp_model = config.get_amplitude()
dg = amp_model.decay_group
config.set_params(params_dict)

per_chain = {}
for chain_idx in range(13):
    dg.set_used_chains([chain_idx])
    amp = dg.get_amp(phsp_variables).numpy().reshape(-1)
    per_chain[str(chain_idx)] = {
        "re": np.real(amp).tolist(),
        "im": np.imag(amp).tolist(),
    }

dg.set_used_chains(list(range(13)))
total_amp = dg.get_amp(phsp_variables).numpy().reshape(-1)

payload = {
    "seed": seed,
    "p4": sampled_p4,
    "per_chain": per_chain,
    "total_re": np.real(total_amp).tolist(),
    "total_im": np.imag(total_amp).tolist(),
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
"""
    write(path, script)
end

function fetch_tfpwa_crosscheck_4b(n::Int, seed::Int)
    analysis_dir = joinpath(repo_root, "archive", "investigation", "Analysis")
    python_exe = resolve_tfpwa_python()
    mktemp() do output_path, io
        close(io)
        mktemp() do script_path, script_io
            close(script_io)
            write_crosscheck_4b_python_script(script_path)
            run(`$python_exe $script_path $repo_root $analysis_dir $output_path $n $seed`)
            return JSON.parsefile(output_path)
        end
    end
end

chain_idx(name::String) = resonance_chain_idx[name]

function fourvector_columns(prefix::String, p4::Vector{Float64})
    return (
        Symbol(prefix * "_px") => p4[2],
        Symbol(prefix * "_py") => p4[3],
        Symbol(prefix * "_pz") => p4[4],
        Symbol(prefix * "_E") => p4[1],
    )
end

function build_crosscheck_4b_table(n::Int, seed::Int)
    output_data = fetch_tfpwa_crosscheck_4b(n, seed)
    p4_data = output_data["p4"]
    rows = NamedTuple[]

    for idx in 1:n
        sampled_p4 = Dict(name => Float64.(p4_data[name][idx]) for name in ["D", "K", "D0", "pi"])
        ctx = event_context(sampled_p4)

        row_pairs = Pair{Symbol, Float64}[]
        push!(row_pairs, :event => idx)
        push!(row_pairs, :seed => Float64(seed))
        push!(row_pairs, :weight => 1.0)
        append!(row_pairs, fourvector_columns("D0", sampled_p4["D0"]))
        append!(row_pairs, fourvector_columns("pip", sampled_p4["pi"]))
        append!(row_pairs, fourvector_columns("Dm", sampled_p4["D"]))
        append!(row_pairs, fourvector_columns("Kp", sampled_p4["K"]))
        push!(row_pairs, :m2_D0pi => mass(ctx.P_Dst)^2)

        julia_total = 0.0 + 0.0im
        for name in all_resonance_names
            tfpwa = Complex(
                Float64(output_data["per_chain"][string(chain_idx(name))]["re"][idx]),
                Float64(output_data["per_chain"][string(chain_idx(name))]["im"][idx]),
            )
            julia = selected_cd_amplitude(ctx, name)
            julia_total += julia
            push!(row_pairs, Symbol(name * "_tfpwa_re") => real(tfpwa))
            push!(row_pairs, Symbol(name * "_tfpwa_im") => imag(tfpwa))
            push!(row_pairs, Symbol(name * "_julia_re") => real(julia))
            push!(row_pairs, Symbol(name * "_julia_im") => imag(julia))
        end

        tfpwa_total = Complex(
            Float64(output_data["total_re"][idx]),
            Float64(output_data["total_im"][idx]),
        )
        push!(row_pairs, :total_tfpwa_re => real(tfpwa_total))
        push!(row_pairs, :total_tfpwa_im => imag(tfpwa_total))
        push!(row_pairs, :total_julia_re => real(julia_total))
        push!(row_pairs, :total_julia_im => imag(julia_total))

        push!(rows, (; row_pairs...))
        idx % 100 == 0 && println("  assembled event ", idx, " / ", n)
    end

    return DataFrame(rows)
end

function max_amplitude_mismatch(df::DataFrame)
    max_abs = 0.0
    for name in all_resonance_names
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

println("Generate crosscheck_4b.arrow")
println("==========================")
println("Events: ", n_crosscheck_events)
println("Seed:   ", crosscheck_seed)
println("Output: ", output_path)
println()

df = build_crosscheck_4b_table(n_crosscheck_events, crosscheck_seed)
max_abs, total_delta = max_amplitude_mismatch(df)

mkpath(dirname(output_path))
Arrow.write(output_path, df)

println()
println(@sprintf("Saved %d events to %s", nrow(df), output_path))
println(@sprintf("Max per-event |A_julia - A_tfpwa| = %.6e", max_abs))
println(@sprintf("Mean total |delta|               = %.6e", mean(total_delta)))
max_abs <= 1e-9 || error("Crosscheck generation mismatch too large: ", max_abs)
println("PASS: stored Julia and TF-PWA amplitudes agree at generation time.")
