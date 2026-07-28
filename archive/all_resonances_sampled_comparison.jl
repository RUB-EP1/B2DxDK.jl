# Change only this line to test one active resonance, or use "all"
# for the coherent sum of all implemented resonances:
const selected_resonance = "all"
const n_events = 50_000
#
# Valid choices:
# "all",
# "X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)",
# "X(3993)", "Psi(4040)", "X(4300)", "NR(0-)SPp",
# "NR(1.)PSp", "NR(0-)SPm", "NR(1-)PPm", "X0(2900)", "X1(2900)"

using CascadeDecays
using FourVectors
using HadronicLineshapes
using JSON
using Plots
using Printf
using Statistics
import ThreeBodyDecays
using ThreeBodyDecays: Recoupling, RecouplingLS, VertexFunction

"""
    MissingParticleTwoPhaseLS(two_ls)

Workaround recoupling that applies the Jacob–Wick particle-2 phase a second time so
it cancels CascadeDecays' built-in factor in `_vertex_coupling_value`. TF-PWA **omits**
this phase (it is missing from that implementation, not a harmless convention difference).
"""
struct MissingParticleTwoPhaseLS <: Recoupling
    two_ls::Tuple{Int,Int}
end

function ThreeBodyDecays.amplitude(cs::MissingParticleTwoPhaseLS, helicities, spins)
    _, _, two_j2 = spins
    _, two_lambda2 = helicities
    exponent_num = two_j2 - two_lambda2
    iseven(exponent_num) || error("particle-2 phase requires two_j2 - two_lambda2 to be even")
    phase = isodd(div(exponent_num, 2)) ? -1 : 1
    return phase * ThreeBodyDecays.amplitude(RecouplingLS(cs.two_ls), helicities, spins)
end

const nominal_mass = Dict(
    "Bp" => 5.27934,
    "D" => 1.86965,
    "K" => 0.493677,
    "D0" => 1.86483,
    "pi" => 0.13957039,
    "Dst" => 2.01026,
    "X0(2900)" => 2.866,
    "X1(2900)" => 2.904,
    "NR(0-)SPp" => 4.35,
    "NR(1.)PSp" => 4.35,
    "NR(0-)SPm" => 4.35,
    "NR(1-)PPm" => 4.35,
)

const resonance_info = Dict(
    "X(3872)" => (chain_idx = 0, model = "C(BWR_LS), D*D l=0 and l=2, below-threshold q0 ad-hoc"),
    "X(3915)(0-)" => (chain_idx = 1, model = "C(BWR), D*D l=1"),
    "chi(c2)(3930)" => (chain_idx = 2, model = "C(BWR), D*D l=2"),
    "X(3940)(1.)" => (chain_idx = 3, model = "C(BWR_LS), D*D l=0 and l=2"),
    "X(3993)" => (chain_idx = 4, model = "C(BWR_LS), D*D l=0 and l=2"),
    "Psi(4040)" => (chain_idx = 5, model = "C(BWR), D*D l=1"),
    "X(4300)" => (chain_idx = 6, model = "C(BWR_LS), D*D l=0 and l=2"),
    "NR(0-)SPp" => (chain_idx = 7, model = "C(New), D*D l=1"),
    "NR(1.)PSp" => (chain_idx = 8, model = "C(one), D*D l=0"),
    "NR(0-)SPm" => (chain_idx = 9, model = "C(one), D*D l=1"),
    "NR(1-)PPm" => (chain_idx = 10, model = "C(one), D*D l=1"),
    "X0(2900)" => (chain_idx = 11, model = "C2(BWR), DK l=0, Bp root L/S=(1,1)"),
    "X1(2900)" => (chain_idx = 12, model = "C2(BWR), DK l=1, Bp root L/S=(0,0),(1,1),(2,2)"),
)

selected_resonance != "all" && !haskey(resonance_info, selected_resonance) &&
    error("Unknown selected_resonance=$(selected_resonance).")

const dstd_resonance_names = [
    "X(3872)",
    "X(3915)(0-)",
    "chi(c2)(3930)",
    "X(3940)(1.)",
    "X(3993)",
    "Psi(4040)",
    "X(4300)",
    "NR(0-)SPp",
    "NR(1.)PSp",
    "NR(0-)SPm",
    "NR(1-)PPm",
]

const all_resonance_names = [
    dstd_resonance_names...,
    "X0(2900)",
    "X1(2900)",
]

const topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))

params_path = normpath(joinpath(@__DIR__, "..", "Analysis", "final_params_full.json"))
params = JSON.parsefile(params_path)["value"]
for name in ["X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)", "X(3993)", "Psi(4040)", "X(4300)", "X0(2900)", "X1(2900)"]
    haskey(params, name * "_mass") && (nominal_mass[name] = Float64(params[name * "_mass"]))
end

param_real(key) = Float64(params[key])
param_complex(key) = param_real(key * "r") * cis(param_real(key * "i"))

tfpwa_breakup(m0, m1, m2) =
    sqrt(complex(((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)))

mismatch_factor(l, d, m0, m1, m2) = 1 / BlattWeisskopf{l}(d)(m0^2, m1^2, m2^2)

function event_context(sampled_p4)
    pDminus = FourVector(sampled_p4["D"][2], sampled_p4["D"][3], sampled_p4["D"][4]; E = sampled_p4["D"][1])
    pD0 = FourVector(sampled_p4["D0"][2], sampled_p4["D0"][3], sampled_p4["D0"][4]; E = sampled_p4["D0"][1])
    pKplus = FourVector(sampled_p4["K"][2], sampled_p4["K"][3], sampled_p4["K"][4]; E = sampled_p4["K"][1])
    piplus = FourVector(sampled_p4["pi"][2], sampled_p4["pi"][3], sampled_p4["pi"][4]; E = sampled_p4["pi"][1])
    objs = (pD0, piplus, pDminus, pKplus)
    P_Dst = pD0 + piplus
    P_R = P_Dst + pDminus
    P_B = P_R + pKplus
    system = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., mass(P_B)^2))
    x = cascade_kinematics(topology, system, objs)
    return (
        sampled_p4 = sampled_p4,
        pDminus = pDminus,
        pD0 = pD0,
        pKplus = pKplus,
        piplus = piplus,
        P_Dst = P_Dst,
        P_R = P_R,
        P_B = P_B,
        system = system,
        x = x,
    )
end

function dk_event_context(sampled_p4)
    pDminus = FourVector(sampled_p4["D"][2], sampled_p4["D"][3], sampled_p4["D"][4]; E = sampled_p4["D"][1])
    pD0 = FourVector(sampled_p4["D0"][2], sampled_p4["D0"][3], sampled_p4["D0"][4]; E = sampled_p4["D0"][1])
    pKplus = FourVector(sampled_p4["K"][2], sampled_p4["K"][3], sampled_p4["K"][4]; E = sampled_p4["K"][1])
    piplus = FourVector(sampled_p4["pi"][2], sampled_p4["pi"][3], sampled_p4["pi"][4]; E = sampled_p4["pi"][1])
    objs = (pD0, piplus, pDminus, pKplus)
    P_Dst = pD0 + piplus
    P_DK = pDminus + pKplus
    P_B = P_Dst + P_DK
    system = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., mass(P_B)^2))
    x = cascade_kinematics(dk_topology, system, objs)
    return (
        pDminus = pDminus,
        pD0 = pD0,
        pKplus = pKplus,
        piplus = piplus,
        P_Dst = P_Dst,
        P_DK = P_DK,
        P_B = P_B,
        system = system,
        x = x,
    )
end

function resolve_tfpwa_python()
    candidates = String[]
    haskey(ENV, "TFPWA_PYTHON") && push!(candidates, ENV["TFPWA_PYTHON"])
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

function write_tfpwa_sampling_script(path::String, chain_idx::Int)
    script = """
import json
import os
import sys

import numpy as np
import tensorflow as tf

repo_root, analysis_dir, output_path, n_events = sys.argv[1:5]
n_events = int(n_events)
chain_spec = sys.argv[5]
sys.path.insert(0, os.path.join(repo_root, "tf-pwa"))
sys.path.insert(0, analysis_dir)
os.chdir(analysis_dir)

import extra_amp
from tf_pwa.config_loader import ConfigLoader
with open("final_params_full.json", "r", encoding="utf-8") as f:
    params_dict = json.load(f)["value"]

seed = int(np.random.default_rng().integers(0, 2**31 - 1))
np.random.seed(seed)
tf.random.set_seed(seed)

config = ConfigLoader("config_a.yml")
phsp_p = config.generate_phsp_p(n_events)
p4_tfpwa = {particle: tf.constant(np.asarray(p4), dtype=tf.float64) for particle, p4 in phsp_p.items()}
sampled_p4 = {particle.name: np.asarray(p4).tolist() for particle, p4 in phsp_p.items()}
phsp_variables = config.data.cal_angle(p4_tfpwa)
phsp_variables["c"] = np.full(n_events, -1.0)

amp_model = config.get_amplitude()
dg = amp_model.decay_group
config.set_params(params_dict)
if chain_spec == "all_dstd":
    dg.set_used_chains(list(range(11)))
elif chain_spec == "all":
    dg.set_used_chains(list(range(13)))
else:
    dg.set_used_chains([int(chain_spec)])
amp = dg.get_amp(phsp_variables).numpy().reshape(-1)

payload = {
    "seed": seed,
    "p4": sampled_p4,
    "tfpwa_amplitude_re": np.real(amp).tolist(),
    "tfpwa_amplitude_im": np.imag(amp).tolist(),
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
"""
    write(path, script)
end

function sample_events_and_tfpwa_reference(n::Int, chain_spec)
    repo_root = normpath(joinpath(@__DIR__, ".."))
    analysis_dir = joinpath(repo_root, "Analysis")
    python_exe = resolve_tfpwa_python()
    output_data = mktemp() do output_path, io
        close(io)
        mktemp() do script_path, script_io
            close(script_io)
            write_tfpwa_sampling_script(script_path, 0)
            run(`$(python_exe) $(script_path) $(repo_root) $(analysis_dir) $(output_path) $(n) $(chain_spec)`)
            JSON.parsefile(output_path)
        end
    end
    p4_data = output_data["p4"]
    sampled_events = [Dict(name => Float64.(p4_data[name][idx]) for name in ["D", "K", "D0", "pi"])
                      for idx in 1:length(p4_data["D"])]
    tfpwa_amp = ComplexF64.(Float64.(output_data["tfpwa_amplitude_re"]), Float64.(output_data["tfpwa_amplitude_im"]))
    return Int(output_data["seed"]), sampled_events, tfpwa_amp
end

bwr_lineshape(ctx, m0, width, l, sign) = begin
    q0 = real(tfpwa_breakup(m0, nominal_mass["Dst"], nominal_mass["D"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = m0 * width / (2q0) * m0 / ff(q0)^2
    # Event-specific alternative:
    # sign * MultichannelBreitWigner(m0, [(; gsq, ma = mass(ctx.P_Dst), mb = mass(ctx.pDminus), l, d = 3.0)])
    sign * MultichannelBreitWigner(m0, [(; gsq, ma = nominal_mass["Dst"], mb = nominal_mass["D"], l, d = 3.0)])
end

function ad_hoc_mass(m0, m_min, m_max)
    k = (m_max - m_min) / 2
    return k * (1 + tanh((2m0 - (m_max + m_min)) / k / 4)) + m_min
end

function bwr_ls_lineshapes(ctx, name, sign; below_threshold = false)
    m0 = nominal_mass[name]
    q0_mass = below_threshold ? ad_hoc_mass(m0, nominal_mass["Dst"] + nominal_mass["D"], nominal_mass["Bp"] - nominal_mass["K"]) : m0
    q0 = real(tfpwa_breakup(q0_mass, nominal_mass["Dst"], nominal_mass["D"]))
    gamma0 = cos(param_real(name * "_theta0"))
    gamma2 = sin(param_real(name * "_theta0"))
    ff0 = BlattWeisskopf{0}(3.0)
    ff2 = BlattWeisskopf{2}(3.0)
    # Event-specific alternatives:
    # channels = [
    #     (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma0^2 / ff0(q0)^2,
    #        ma = mass(ctx.P_Dst), mb = mass(ctx.pDminus), l = 0, d = 3.0),
    #     (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma2^2 / ff2(q0)^2,
    #        ma = mass(ctx.P_Dst), mb = mass(ctx.pDminus), l = 2, d = 3.0),
    # ]
    channels = [
        (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma0^2 / ff0(q0)^2,
           ma = nominal_mass["Dst"], mb = nominal_mass["D"], l = 0, d = 3.0),
        (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma2^2 / ff2(q0)^2,
           ma = nominal_mass["Dst"], mb = nominal_mass["D"], l = 2, d = 3.0),
    ]
    bw = sign * MultichannelBreitWigner(m0, channels)
    # Event-specific alternative:
    # breakup_from_sigma = sigma -> tfpwa_breakup(sqrt(sigma), mass(ctx.P_Dst), mass(ctx.pDminus))
    breakup_from_sigma = sigma -> tfpwa_breakup(sqrt(sigma), nominal_mass["Dst"], nominal_mass["D"])
    return bw * gamma0, bw * (ff2(breakup_from_sigma) * (gamma2 / ff2(q0)))
end

function chain_amplitude(ctx, lineshape, two_j, root_two_ls, decay_two_ls; root_l = nothing, decay_l = nothing)
    root_vertex = root_l === nothing ?
        VertexFunction(RecouplingLS(root_two_ls)) :
        VertexFunction(RecouplingLS(root_two_ls), BlattWeisskopf{root_l}(3.0))
    decay_vertex = decay_l === nothing ?
        VertexFunction(RecouplingLS(decay_two_ls)) :
        VertexFunction(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(3.0))
    chain = CascadeDecays.DecayChain(
        topology;
        propagators = (
            (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0.0im)),
            ((1, 2), 3) => (two_j = two_j, lineshape = lineshape),
        ),
        vertices = (
            (((1, 2), 3), 4) => root_vertex,
            ((1, 2), 3) => decay_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, ctx.system, ctx.x, (0, 0, 0, 0, 0))
end

function dk_chain_amplitude(ctx, lineshape, two_j, root_two_ls, dk_two_ls; root_l = nothing, dk_l = nothing, missing_particle2_phase = false)
    root_recoupling = missing_particle2_phase ? MissingParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    root_vertex = root_l === nothing ?
        VertexFunction(root_recoupling) :
        VertexFunction(root_recoupling, BlattWeisskopf{root_l}(3.0))
    dk_vertex = dk_l === nothing ?
        VertexFunction(RecouplingLS(dk_two_ls)) :
        VertexFunction(RecouplingLS(dk_two_ls), BlattWeisskopf{dk_l}(3.0))
    chain = CascadeDecays.DecayChain(
        dk_topology;
        propagators = (
            (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0.0im)),
            (3, 4) => (two_j = two_j, lineshape = lineshape),
        ),
        vertices = (
            ((1, 2), (3, 4)) => root_vertex,
            (3, 4) => dk_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, ctx.system, ctx.x, (0, 0, 0, 0, 0))
end

function x2900_bwr_lineshape(ctx, name, l)
    q0 = real(tfpwa_breakup(nominal_mass[name], nominal_mass["D"], nominal_mass["K"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = nominal_mass[name] * param_real(name * "_width") / (2q0) * nominal_mass[name] / ff(q0)^2
    # Event-specific alternative:
    # return MultichannelBreitWigner(
    #     nominal_mass[name],
    #     [(; gsq, ma = mass(ctx.pDminus), mb = mass(ctx.pKplus), l, d = 3.0)],
    # )
    return MultichannelBreitWigner(
        nominal_mass[name],
        [(; gsq, ma = nominal_mass["D"], mb = nominal_mass["K"], l, d = 3.0)],
    )
end

function selected_cd_amplitude(ctx, name::String)
    if name == "X(3872)"
        l0, l2 = bwr_ls_lineshapes(ctx, "X(3872)", -1.0 + 0im; below_threshold = true)
        root_ff = BlattWeisskopf{1}(3.0)
        root_mdep = root_ff(tfpwa_breakup(mass(ctx.P_B), mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(tfpwa_breakup(nominal_mass["Bp"], nominal_mass["X(3872)"], nominal_mass["K"]))
        amp_l0 = root_mdep * chain_amplitude(ctx, l0, 2, (2, 2), (0, 2))
        amp_l2 = root_mdep * chain_amplitude(ctx, l2, 2, (2, 2), (4, 2))
        return param_complex("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0") *
               (amp_l0 + param_complex("X(3872)->Dst.D_g_ls_1") * amp_l2)
    elseif name == "X(3915)(0-)"
        raw = chain_amplitude(
            ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 1, -1.0 + 0im),
            0, (0, 0), (2, 2); root_l = 0, decay_l = 1,
        )
        correction = mismatch_factor(0, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) *
                     mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "chi(c2)(3930)"
        raw = chain_amplitude(
            ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 2, -1.0 + 0im),
            4, (4, 4), (4, 2); root_l = 2, decay_l = 2,
        )
        correction = mismatch_factor(2, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) *
                     mismatch_factor(2, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "X(3940)(1.)" || name == "X(3993)" || name == "X(4300)"
        sign = name == "X(3993)" ? -1.0 + 0im : 1.0 + 0im
        l0, l2 = bwr_ls_lineshapes(ctx, name, sign)
        root_ff = BlattWeisskopf{1}(3.0)
        root_mdep = root_ff(tfpwa_breakup(mass(ctx.P_B), mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(tfpwa_breakup(nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]))
        amp_l0 = root_mdep * chain_amplitude(ctx, l0, 2, (2, 2), (0, 2))
        amp_l2 = root_mdep * chain_amplitude(ctx, l2, 2, (2, 2), (4, 2))
        total_key = name == "X(3940)(1.)" ?
            "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0" :
            "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
        return param_complex(total_key) * (amp_l0 + param_complex("$(name)->Dst.D_g_ls_1") * amp_l2)
    elseif name == "Psi(4040)"
        raw = chain_amplitude(
            ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 1, 1.0 + 0im),
            2, (2, 2), (2, 2); root_l = 1, decay_l = 1,
        )
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) *
                     mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(0-)SPp"
        alpha = param_real("NR(0-)SPp_alpha")
        beta = param_real("NR(0-)SPp_beta")
        nr_factor = -exp(-(alpha + 1im * beta) * (mass(ctx.P_R)^2 - nominal_mass["NR(0-)SPp"]^2))
        raw = chain_amplitude(ctx, ConstantLineshape(nr_factor), 0, (0, 0), (2, 2); decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["NR(0-)SPp"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(1.)PSp"
        raw = chain_amplitude(ctx, ConstantLineshape(-1.0 + 0im), 2, (2, 2), (0, 2); root_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["NR(1.)PSp"], nominal_mass["K"])
        return param_complex("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(0-)SPm"
        raw = chain_amplitude(ctx, ConstantLineshape(1.0 + 0im), 0, (0, 0), (2, 2); decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["NR(0-)SPm"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(1-)PPm"
        raw = chain_amplitude(ctx, ConstantLineshape(1.0 + 0im), 2, (2, 2), (2, 2); root_l = 1, decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["NR(1-)PPm"], nominal_mass["K"]) *
                     mismatch_factor(1, 3.0, nominal_mass["NR(1-)PPm"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "X0(2900)"
        dk_ctx = dk_event_context(ctx.sampled_p4)
        raw = dk_chain_amplitude(
            dk_ctx, x2900_bwr_lineshape(dk_ctx, name, 0),
            0, (2, 2), (0, 0); root_l = 1, dk_l = 0,
        )
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) *
                     mismatch_factor(0, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        return param_complex("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0") * raw * correction
    elseif name == "X1(2900)"
        dk_ctx = dk_event_context(ctx.sampled_p4)
        lineshape = x2900_bwr_lineshape(dk_ctx, name, 1)
        raw_l0 = dk_chain_amplitude(dk_ctx, lineshape, 2, (0, 0), (2, 0); root_l = 0, dk_l = 1, missing_particle2_phase = true)
        raw_l1 = dk_chain_amplitude(dk_ctx, lineshape, 2, (2, 2), (2, 0); root_l = 1, dk_l = 1, missing_particle2_phase = true)
        raw_l2 = dk_chain_amplitude(dk_ctx, lineshape, 2, (4, 4), (2, 0); root_l = 2, dk_l = 1, missing_particle2_phase = true)
        correction_l0 = mismatch_factor(0, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) *
                        mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        correction_l1 = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) *
                        mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        correction_l2 = mismatch_factor(2, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) *
                        mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        coherent = raw_l0 * correction_l0 +
                   param_complex("Bp->X1(2900).Dst_g_ls_1") * raw_l1 * correction_l1 +
                   param_complex("Bp->X1(2900).Dst_g_ls_2") * raw_l2 * correction_l2
        return param_complex("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0") * coherent
    end
    error("No branch implemented for $(name).")
end

all_resonance_cd_amplitude(ctx) = sum(selected_cd_amplitude(ctx, name) for name in all_resonance_names)

function save_scan(path::String, rows)
    header = ["event", "m2_D0pi", "m2_D0piD", "m2_DK", "m2_DstK", "tfpwa_re", "tfpwa_im",
              "cd_re", "cd_im", "rel_delta_re", "rel_delta_im", "abs_delta"]
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            values = Any[row.event, row.m2_dst, row.m2_r, row.m2_dk, row.m2_dstk, real(row.tfpwa_amp), imag(row.tfpwa_amp),
                         real(row.cd_amp), imag(row.cd_amp), row.rel_delta_re, row.rel_delta_im, abs(row.delta)]
            println(io, join(string.(values), '\t'))
        end
    end
end

function save_fit_fractions(path::String, component_names, component_amps_by_event)
    coherent_norm = sum(abs2(sum(component_amps)) for component_amps in component_amps_by_event)
    component_norms = [
        sum(abs2(component_amps[idx]) for component_amps in component_amps_by_event)
        for idx in eachindex(component_names)
    ]
    fit_fractions = component_norms ./ coherent_norm
    incoherent_norm = sum(component_norms)
    interference_fraction = (coherent_norm - incoherent_norm) / coherent_norm

    open(path, "w") do io
        println(io, join(["component", "sum_abs2_component", "fit_fraction", "fit_fraction_percent"], '\t'))
        for (name, norm, fraction) in zip(component_names, component_norms, fit_fractions)
            println(io, join(string.((name, norm, fraction, 100fraction)), '\t'))
        end
        println(io, join(string.(("interference", coherent_norm - incoherent_norm,
                                 interference_fraction, 100interference_fraction)), '\t'))
    end

    println()
    println("CascadeDecays fit fractions:")
    for (name, fraction) in zip(component_names, fit_fractions)
        println(@sprintf("  %-16s %.6e  (%.4f%%)", name, fraction, 100fraction))
    end
    println(@sprintf("  %-16s %.6e  (%.4f%%)", "interference", interference_fraction, 100interference_fraction))
    return fit_fractions, interference_fraction
end

function binned_mean_grid(rows, values; nbins::Int = 100)
    xs = [row.m2_r for row in rows]
    ys = [row.m2_dk for row in rows]
    xedges = collect(range(minimum(xs), maximum(xs); length = nbins + 1))
    yedges = collect(range(minimum(ys), maximum(ys); length = nbins + 1))
    sums = zeros(Float64, nbins, nbins)
    counts = zeros(Int, nbins, nbins)
    for (x, y, z) in zip(xs, ys, values)
        ix = clamp(searchsortedlast(xedges, x), 1, nbins)
        iy = clamp(searchsortedlast(yedges, y), 1, nbins)
        if isfinite(z)
            sums[ix, iy] += z
            counts[ix, iy] += 1
        end
    end
    grid = fill(NaN, nbins, nbins)
    for ix in 1:nbins, iy in 1:nbins
        counts[ix, iy] > 0 && (grid[ix, iy] = sums[ix, iy] / counts[ix, iy])
    end
    return xedges, yedges, grid
end

function amplitude_comparison_plot(path::String, rows; nbins::Int = 100)
    tf_values = [abs2(row.tfpwa_amp) for row in rows]
    cd_values = [abs2(row.cd_amp) for row in rows]
    reference_magnitudes = abs.(getproperty.(rows, :tfpwa_amp))
    denominator_threshold = eps(Float64) * max(1.0, maximum(reference_magnitudes))
    relative_values = [
        reference_magnitudes[idx] > denominator_threshold ?
            abs(rows[idx].delta) / reference_magnitudes[idx] : NaN
        for idx in eachindex(rows)
    ]

    xedges, yedges, tf_grid = binned_mean_grid(rows, tf_values; nbins)
    _, _, cd_grid = binned_mean_grid(rows, cd_values; nbins)
    _, _, relative_grid = binned_mean_grid(rows, relative_values; nbins)

    positive = vcat(vec(tf_grid[(isfinite.(tf_grid)) .& (tf_grid .> 0)]),
                    vec(cd_grid[(isfinite.(cd_grid)) .& (cd_grid .> 0)]))
    intensity_min = max(quantile(positive, 0.02), floatmin(Float64))
    intensity_max = max(quantile(positive, 0.98), 10intensity_min)

    finite_relative_values = relative_values[isfinite.(relative_values)]
    mean_relative_difference = mean(finite_relative_values)
    relative_scale_exponent = mean_relative_difference > 0 ? floor(Int, log10(mean_relative_difference)) : 0
    relative_scale = 10.0^relative_scale_exponent
    scaled_relative_grid = relative_grid ./ relative_scale
    positive_relative = scaled_relative_grid[(isfinite.(scaled_relative_grid)) .& (scaled_relative_grid .> 0)]
    relative_min = max(minimum(positive_relative), floatmin(Float64))
    relative_max = max(maximum(positive_relative), 10relative_min)
    relative_ticks = 10.0 .^ range(log10(relative_min), log10(relative_max); length = 5)
    log_relative_grid = map(scaled_relative_grid) do value
        isfinite(value) && value > 0 ? log10(value) : NaN
    end

    centers_x = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    centers_y = (yedges[1:end-1] .+ yedges[2:end]) ./ 2
    common_intensity = (
        c = :viridis,
        clim = (intensity_min, intensity_max),
        colorbar_scale = :log10,
        background_color = :white,
        background_color_inside = :white,
        grid = false,
        titlefontsize = 14,
        guidefontsize = 14,
        tickfontsize = 12,
        colorbar_tickfontsize = 12,
        bottom_margin = 12Plots.mm,
    )
    p1 = heatmap(
        centers_x, centers_y, tf_grid';
        title = "Original TF-PWA: mean |A|²",
        xlabel = "m²(D*D) [GeV²]",
        ylabel = "m²(DK) [GeV²]",
        colorbar = false,
        common_intensity...,
        left_margin = 12Plots.mm,
    )
    p2 = heatmap(
        centers_x, centers_y, cd_grid';
        title = "CascadeDecays: mean |A|²",
        xlabel = "m²(D*D) [GeV²]",
        yticks = false,
        colorbar = false,
        common_intensity...,
    )
    p3 = heatmap(
        centers_x, centers_y, log_relative_grid';
        title = "Mean |A_CD - A_TF| / |A_TF|\n(values divided by 10^$(relative_scale_exponent))",
        xlabel = "m²(D*D) [GeV²]",
        yticks = false,
        c = :magma,
        clim = log10.((relative_min, relative_max)),
        colorbar = false,
        background_color = :white,
        background_color_inside = :white,
        grid = false,
        titlefontsize = 14,
        guidefontsize = 14,
        tickfontsize = 12,
        colorbar_tickfontsize = 12,
        bottom_margin = 12Plots.mm,
    )

    # GR does not honor custom labels on native colorbars, so draw both scales explicitly.
    intensity_colorbar_values = collect(range(log10(intensity_min), log10(intensity_max); length = 256))
    intensity_tick_exponents = collect(ceil(Int, log10(intensity_min)):floor(Int, log10(intensity_max)))
    p4 = heatmap(
        [0.0], intensity_colorbar_values, reshape(intensity_colorbar_values, :, 1);
        c = :viridis,
        clim = log10.((intensity_min, intensity_max)),
        colorbar = false,
        xaxis = false,
        yticks = (Float64.(intensity_tick_exponents), [@sprintf("10^%d", exponent) for exponent in intensity_tick_exponents]),
        ymirror = true,
        grid = false,
        framestyle = :box,
        tickfontsize = 12,
        top_margin = 9Plots.mm,
        bottom_margin = 12Plots.mm,
        right_margin = 0Plots.mm,
    )
    relative_colorbar_values = collect(range(log10(relative_min), log10(relative_max); length = 256))
    p5 = heatmap(
        [0.0], relative_colorbar_values, reshape(relative_colorbar_values, :, 1);
        c = :magma,
        clim = log10.((relative_min, relative_max)),
        colorbar = false,
        xaxis = false,
        yticks = (log10.(relative_ticks), [@sprintf("%.2g", tick) for tick in relative_ticks]),
        ymirror = true,
        grid = false,
        framestyle = :box,
        tickfontsize = 12,
        top_margin = 9Plots.mm,
        bottom_margin = 12Plots.mm,
        right_margin = 0Plots.mm,
    )
    comparison_layout = @layout [a{0.30w} b{0.30w} c{0.05w} d{0.30w} e{0.05w}]
    comparison = plot(
        p1, p2, p4, p3, p5;
        layout = comparison_layout,
        size = (2250, 750),
        dpi = 300,
        margin = 2Plots.mm,
    )
    savefig(comparison, path)
    println("Relative-difference color scale: values divided by 10^", relative_scale_exponent)
end

function weighted_histogram(xs, weights; nbins::Int = 80)
    lo, hi = extrema(xs)
    edges = collect(range(lo, hi; length = nbins + 1))
    sums = zeros(Float64, nbins)
    for (x, w) in zip(xs, weights)
        if isfinite(x) && isfinite(w)
            idx = x == hi ? nbins : clamp(searchsortedlast(edges, x), 1, nbins)
            sums[idx] += w
        end
    end
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, sums
end

function save_resonance_amp2_histograms(plots_dir::String, rows, resonance_amps_by_event; nbins::Int = 80)
    hist_dir = joinpath(plots_dir, "amplitude_histograms")
    mkpath(hist_dir)
    mass_specs = [
        (field = :m2_r, label = "m(D*D) [GeV]", filename = "amp2_hist_m_DstD.png"),
        (field = :m2_dk, label = "m(DK) [GeV]", filename = "amp2_hist_m_DK.png"),
        (field = :m2_dstk, label = "m(D*K) [GeV]", filename = "amp2_hist_m_DstK.png"),
    ]

    for spec in mass_specs
        xs = [sqrt(getproperty(row, spec.field)) for row in rows]
        plt = plot(
            xlabel = spec.label,
            ylabel = "sum |A|^2 per bin",
            title = "Resonance |A|^2 histograms vs $(spec.label)",
            background_color = :white,
            background_color_inside = :white,
            legend = :outerright,
            size = (1500, 850),
            right_margin = 18Plots.mm,
        )
        for (idx, name) in enumerate(all_resonance_names)
            weights = [abs2(amps[idx]) for amps in resonance_amps_by_event]
            centers, sums = weighted_histogram(xs, weights; nbins)
            plot!(plt, centers, sums; label = name, linewidth = 1.0, alpha = 0.65)
        end
        total_weights = [abs2(sum(amps)) for amps in resonance_amps_by_event]
        centers, sums = weighted_histogram(xs, total_weights; nbins)
        plot!(plt, centers, sums; label = "total", color = :black, linewidth = 3.0)
        savefig(plt, joinpath(hist_dir, spec.filename))
    end
    println("Saved resonance |A|^2 histograms: ", hist_dir)
    return hist_dir
end

function run_scan()
    info = resonance_info[selected_resonance]
    println("Isolated single-resonance random-event D*D comparison")
    println("=====================================================")
    println("Selected resonance: ", selected_resonance)
    println("TF-PWA chain index: ", info.chain_idx)
    println("Model / LS setup:   ", info.model)
    println("Requested events:   ", n_events)
    println()

    seed, sampled_events, tfpwa_amps = sample_events_and_tfpwa_reference(n_events, info.chain_idx)
    println("TF-PWA phase-space seed: ", seed)
    println("Computing CascadeDecays amplitudes event by event...")

    rows = NamedTuple[]
    for idx in eachindex(sampled_events)
        ctx = event_context(sampled_events[idx])
        cd_amp = selected_cd_amplitude(ctx, selected_resonance)
        tfpwa_amp = tfpwa_amps[idx]
        rel_delta_re = abs(real(tfpwa_amp)) > 1e-12 ? (real(cd_amp) - real(tfpwa_amp)) / real(tfpwa_amp) : NaN
        rel_delta_im = abs(imag(tfpwa_amp)) > 1e-12 ? (imag(cd_amp) - imag(tfpwa_amp)) / imag(tfpwa_amp) : NaN
        push!(rows, (
            event = idx,
            m2_dst = mass(ctx.P_Dst)^2,
            m2_r = mass(ctx.P_R)^2,
            m2_dk = mass(ctx.pDminus + ctx.pKplus)^2,
            m2_dstk = mass(ctx.P_Dst + ctx.pKplus)^2,
            tfpwa_amp = tfpwa_amp,
            cd_amp = cd_amp,
            delta = cd_amp - tfpwa_amp,
            rel_delta_re = rel_delta_re,
            rel_delta_im = rel_delta_im,
        ))
        idx % 1000 == 0 && println("  processed ", idx, " / ", n_events)
    end

    safe_name = replace(replace(replace(selected_resonance, "(" => ""), ")" => ""), "." => "")
    safe_name = replace(replace(safe_name, "+" => "p"), "-" => "m")
    prefix = lowercase(replace(safe_name, " " => "_"))
    plots_dir = joinpath(@__DIR__, "Plots")
    mkpath(plots_dir)
    scan_path = joinpath(@__DIR__, "$(prefix)_single_resonance_scan.txt")
    comparison_path = joinpath(plots_dir, "$(prefix)_single_resonance_amplitude_comparison_dalitz.pdf")

    save_scan(scan_path, rows)
    amplitude_comparison_plot(comparison_path, rows)

    finite_re = [row.rel_delta_re for row in rows if isfinite(row.rel_delta_re)]
    finite_im = [row.rel_delta_im for row in rows if isfinite(row.rel_delta_im)]
    println()
    println("Saved event table: ", scan_path)
    println("Saved plot:        ", comparison_path)
    println(@sprintf("Mean |Delta A|          = %.6e", mean(abs.(getproperty.(rows, :delta)))))
    !isempty(finite_re) && println(@sprintf("Mean relative Delta Re  = %.6e", mean(finite_re)))
    !isempty(finite_im) && println(@sprintf("Mean relative Delta Im  = %.6e", mean(finite_im)))
end

function run_all_resonance_scan()
    println()
    println("All-resonance random-event comparison")
    println("=====================================")
    println("Included CD resonances:")
    for name in all_resonance_names
        println("  ", name)
    end
    println("Requested events: ", n_events)
    println()

    seed, sampled_events, tfpwa_amps = sample_events_and_tfpwa_reference(n_events, "all")
    println("TF-PWA phase-space seed: ", seed)
    println("Computing coherent CascadeDecays sum event by event...")

    rows = NamedTuple[]
    resonance_amps_by_event = Vector{Vector{ComplexF64}}()
    for idx in eachindex(sampled_events)
        ctx = event_context(sampled_events[idx])
        resonance_amps = ComplexF64[selected_cd_amplitude(ctx, name) for name in all_resonance_names]
        cd_amp = sum(resonance_amps)
        tfpwa_amp = tfpwa_amps[idx]
        rel_delta_re = abs(real(tfpwa_amp)) > 1e-12 ? (real(cd_amp) - real(tfpwa_amp)) / real(tfpwa_amp) : NaN
        rel_delta_im = abs(imag(tfpwa_amp)) > 1e-12 ? (imag(cd_amp) - imag(tfpwa_amp)) / imag(tfpwa_amp) : NaN
        push!(resonance_amps_by_event, resonance_amps)
        push!(rows, (
            event = idx,
            m2_dst = mass(ctx.P_Dst)^2,
            m2_r = mass(ctx.P_R)^2,
            m2_dk = mass(ctx.pDminus + ctx.pKplus)^2,
            m2_dstk = mass(ctx.P_Dst + ctx.pKplus)^2,
            tfpwa_amp = tfpwa_amp,
            cd_amp = cd_amp,
            delta = cd_amp - tfpwa_amp,
            rel_delta_re = rel_delta_re,
            rel_delta_im = rel_delta_im,
        ))
        idx % 1000 == 0 && println("  processed ", idx, " / ", n_events)
    end

    plots_dir = joinpath(@__DIR__, "Plots")
    mkpath(plots_dir)
    scan_path = joinpath(@__DIR__, "all_resonances_scan.txt")
    fit_fraction_path = joinpath(@__DIR__, "all_resonances_fit_fractions.txt")
    comparison_path = joinpath(plots_dir, "all_resonances_amplitude_comparison_dalitz.pdf")

    save_scan(scan_path, rows)
    save_fit_fractions(fit_fraction_path, all_resonance_names, resonance_amps_by_event)
    amplitude_comparison_plot(comparison_path, rows)
    save_resonance_amp2_histograms(plots_dir, rows, resonance_amps_by_event)

    finite_re = [row.rel_delta_re for row in rows if isfinite(row.rel_delta_re)]
    finite_im = [row.rel_delta_im for row in rows if isfinite(row.rel_delta_im)]
    rel_abs = [abs(row.delta) / max(abs(row.tfpwa_amp), eps(Float64)) for row in rows]
    println()
    println("Saved all-resonance event table: ", scan_path)
    println("Saved all-resonance fit fractions: ", fit_fraction_path)
    println("Saved all-resonance plot:        ", comparison_path)
    println(@sprintf("Mean |Delta A|                 = %.6e", mean(abs.(getproperty.(rows, :delta)))))
    println(@sprintf("Median relative |Delta A|      = %.6e", median(rel_abs)))
    println(@sprintf("95%% quantile relative |Delta A| = %.6e", quantile(rel_abs, 0.95)))
    !isempty(finite_re) && println(@sprintf("Mean relative Delta Re         = %.6e", mean(finite_re)))
    !isempty(finite_im) && println(@sprintf("Mean relative Delta Im         = %.6e", mean(finite_im)))
end

if selected_resonance == "all"
    run_all_resonance_scan()
else
    run_scan()
end
