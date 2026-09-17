"""
Case (b): PDG Parent Masses + Nominal Lineshapes
- Place 1 (Lineshapes): Only nominal PDG masses
- Place 2 (Parent Masses): Fixed nominal PDG masses
"""

using CascadeDecays, FourVectors, HadronicLineshapes, JSON, Printf
import CascadeDecays.ThreeBodyDecays
using CascadeDecays.ThreeBodyDecays: Recoupling, RecouplingLS, VertexFunction

# Recoupling with particle-2 phase removed for D*K chain
struct RemoveParticleTwoPhaseLS <: Recoupling
    two_ls::Tuple{Int,Int}
end

function ThreeBodyDecays.amplitude(cs::RemoveParticleTwoPhaseLS, helicities, spins)
    _, _, two_j2 = spins
    _, two_lambda2 = helicities
    exponent_num = two_j2 - two_lambda2
    iseven(exponent_num) || error("particle-2 phase requires two_j2 - two_lambda2 to be even")
    phase = isodd(div(exponent_num, 2)) ? -1 : 1
    return phase * ThreeBodyDecays.amplitude(RecouplingLS(cs.two_ls), helicities, spins)
end

# Model constants and topologies
const nominal_mass = Dict(
    "Bp" => 5.27934, "D" => 1.86965, "K" => 0.493677, "D0" => 1.86483,
    "pi" => 0.13957039, "Dst" => 2.01026, "X0(2900)" => 2.866, "X1(2900)" => 2.904,
    "NR(0-)SPp" => 4.35, "NR(1.)PSp" => 4.35, "NR(0-)SPm" => 4.35, "NR(1-)PPm" => 4.35,
)

const dstd_resonance_names = [
    "X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)",
    "X(3993)", "Psi(4040)", "X(4300)", "NR(0-)SPp",
    "NR(1.)PSp", "NR(0-)SPm", "NR(1-)PPm",
]
const all_resonance_names = [dstd_resonance_names..., "X0(2900)", "X1(2900)"]

const topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))

tfpwa_breakup(m0, m1, m2) = sqrt(complex(((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)))
mismatch_factor(l, d, m0, m1, m2) = 1 / BlattWeisskopf{l}(d)(m0^2, m1^2, m2^2)

function ad_hoc_mass(m0, m_min, m_max)
    k = (m_max - m_min) / 2
    return k * (1 + tanh((2m0 - (m_max + m_min)) / k / 4)) + m_min
end

# Event kinematics and cascade systems
function build_event_context(sampled_p4)
    pDminus = FourVector(sampled_p4["D"][2], sampled_p4["D"][3], sampled_p4["D"][4]; E = sampled_p4["D"][1])
    pD0     = FourVector(sampled_p4["D0"][2], sampled_p4["D0"][3], sampled_p4["D0"][4]; E = sampled_p4["D0"][1])
    pKplus  = FourVector(sampled_p4["K"][2], sampled_p4["K"][3], sampled_p4["K"][4]; E = sampled_p4["K"][1])
    piplus  = FourVector(sampled_p4["pi"][2], sampled_p4["pi"][3], sampled_p4["pi"][4]; E = sampled_p4["pi"][1])
    objs = (pD0, piplus, pDminus, pKplus)
    P_Dst = pD0 + piplus
    P_R = P_Dst + pDminus
    P_B = P_R + pKplus
    P_DK = pDminus + pKplus

    # Place 2: Fixed nominal PDG parent masses
    m_B_root = nominal_mass["Bp"]
    m_Dst_int = nominal_mass["Dst"]

    sys_dstd = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., m_B_root^2))
    raw_x_dstd = cascade_kinematics(topology, sys_dstd, objs)
    x_dstd = CascadeDecays.CascadeKinematics(
        topology, sys_dstd;
        internal_masses2 = (m_Dst_int^2, mass(P_R)^2),
        vertex_angles = raw_x_dstd.vertex_angles,
    )

    sys_dk = CascadeSystem((0, 0, 0, 0, 0), (mass.(objs) .^ 2..., m_B_root^2))
    raw_x_dk = cascade_kinematics(dk_topology, sys_dk, objs)
    x_dk = CascadeDecays.CascadeKinematics(
        dk_topology, sys_dk;
        internal_masses2 = (m_Dst_int^2, mass(P_DK)^2),
        vertex_angles = raw_x_dk.vertex_angles,
    )

    return (; sampled_p4, objs, pDminus, pD0, pKplus, piplus,
              P_Dst, P_R, P_B, P_DK, m_B_root, m_Dst_int,
              sys_dstd, x_dstd, sys_dk, x_dk)
end

# Place 1: Only nominal masses in lineshapes
function bwr_lineshape(ctx, m0, width, l, sign)
    q0 = real(tfpwa_breakup(m0, nominal_mass["Dst"], nominal_mass["D"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = m0 * width / (2q0) * m0 / ff(q0)^2
    return sign * MultichannelBreitWigner(m0, [(; gsq, ma = nominal_mass["Dst"], mb = nominal_mass["D"], l, d = 3.0)])
end

function bwr_ls_lineshapes(ctx, name, sign, param_real; below_threshold = false)
    m0 = nominal_mass[name]
    q0_mass = below_threshold ? ad_hoc_mass(m0, nominal_mass["Dst"] + nominal_mass["D"], nominal_mass["Bp"] - nominal_mass["K"]) : m0
    q0 = real(tfpwa_breakup(q0_mass, nominal_mass["Dst"], nominal_mass["D"]))
    gamma0 = cos(param_real(name * "_theta0"))
    gamma2 = sin(param_real(name * "_theta0"))
    ff0 = BlattWeisskopf{0}(3.0)
    ff2 = BlattWeisskopf{2}(3.0)
    channels = [
        (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma0^2 / ff0(q0)^2,
           ma = nominal_mass["Dst"], mb = nominal_mass["D"], l = 0, d = 3.0),
        (; gsq = param_real(name * "_width") * mass(ctx.P_R)^2 / (2q0) * gamma2^2 / ff2(q0)^2,
           ma = nominal_mass["Dst"], mb = nominal_mass["D"], l = 2, d = 3.0),
    ]
    bw = sign * MultichannelBreitWigner(m0, channels)
    breakup_from_sigma = sigma -> tfpwa_breakup(sqrt(sigma), nominal_mass["Dst"], nominal_mass["D"])
    return bw * gamma0, bw * (ff2(breakup_from_sigma) * (gamma2 / ff2(q0)))
end

function x2900_bwr_lineshape(ctx, name, l, param_real)
    q0 = real(tfpwa_breakup(nominal_mass[name], nominal_mass["D"], nominal_mass["K"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = nominal_mass[name] * param_real(name * "_width") / (2q0) * nominal_mass[name] / ff(q0)^2
    return MultichannelBreitWigner(nominal_mass[name], [(; gsq, ma = nominal_mass["D"], mb = nominal_mass["K"], l, d = 3.0)])
end

# Decay chain evaluators
function eval_chain(ctx, lineshape, two_j, root_two_ls, decay_two_ls; root_l = nothing, decay_l = nothing)
    root_vertex = root_l === nothing ? VertexFunction(RecouplingLS(root_two_ls)) : VertexFunction(RecouplingLS(root_two_ls), BlattWeisskopf{root_l}(3.0))
    decay_vertex = decay_l === nothing ? VertexFunction(RecouplingLS(decay_two_ls)) : VertexFunction(RecouplingLS(decay_two_ls), BlattWeisskopf{decay_l}(3.0))
    chain = CascadeDecays.DecayChain(
        topology;
        propagators = (
            (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0im)),
            ((1, 2), 3) => (two_j = two_j, lineshape = lineshape),
        ),
        vertices = (
            (((1, 2), 3), 4) => root_vertex,
            ((1, 2), 3) => decay_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, ctx.sys_dstd, ctx.x_dstd, (0, 0, 0, 0, 0))
end

function eval_dk_chain(ctx, lineshape, two_j, root_two_ls, decay_two_ls; root_l = nothing, dk_l = nothing, remove_root_particle2_phase = false)
    root_rec = remove_root_particle2_phase ? RemoveParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
    root_vertex = root_l === nothing ? VertexFunction(root_rec) : VertexFunction(root_rec, BlattWeisskopf{root_l}(3.0))
    decay_vertex = dk_l === nothing ? VertexFunction(RecouplingLS(decay_two_ls)) : VertexFunction(RecouplingLS(decay_two_ls), BlattWeisskopf{dk_l}(3.0))
    chain = CascadeDecays.DecayChain(
        dk_topology;
        propagators = (
            (1, 2) => (two_j = 2, lineshape = ConstantLineshape(1.0 + 0im)),
            (3, 4) => (two_j = two_j, lineshape = lineshape),
        ),
        vertices = (
            ((1, 2), (3, 4)) => root_vertex,
            (3, 4) => decay_vertex,
            (1, 2) => VertexFunction(RecouplingLS((2, 0))),
        ),
    )
    return CascadeDecays.amplitude(chain, ctx.sys_dk, ctx.x_dk, (0, 0, 0, 0, 0))
end

# Selected resonance amplitudes
function selected_cd_amplitude(ctx, name, param_real, param_complex)
    if name == "X(3872)"
        l0, l2 = bwr_ls_lineshapes(ctx, name, -1.0 + 0im, param_real; below_threshold = true)
        root_ff = BlattWeisskopf{1}(3.0)
        root_mdep = root_ff(tfpwa_breakup(ctx.m_B_root, mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(tfpwa_breakup(nominal_mass["Bp"], nominal_mass["X(3872)"], nominal_mass["K"]))
        amp_l0 = root_mdep * eval_chain(ctx, l0, 2, (2, 2), (0, 2))
        amp_l2 = root_mdep * eval_chain(ctx, l2, 2, (2, 2), (4, 2))
        return param_complex("Bp->X(3872).KX(3872)->Dst.DDst->D0.pi_total_0") * (amp_l0 + param_complex("X(3872)->Dst.D_g_ls_1") * amp_l2)
    elseif name == "X(3915)(0-)"
        raw = eval_chain(ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 1, -1.0 + 0im), 0, (0, 0), (2, 2); root_l = 0, decay_l = 1)
        correction = mismatch_factor(0, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) * mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->X(3915)(0-).KX(3915)(0-)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "chi(c2)(3930)"
        raw = eval_chain(ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 2, -1.0 + 0im), 4, (4, 4), (4, 2); root_l = 2, decay_l = 2)
        correction = mismatch_factor(2, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) * mismatch_factor(2, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->chi(c2)(3930).Kchi(c2)(3930)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "X(3940)(1.)" || name == "X(3993)" || name == "X(4300)"
        sign = name == "X(3993)" ? -1.0 + 0im : 1.0 + 0im
        l0, l2 = bwr_ls_lineshapes(ctx, name, sign, param_real)
        root_ff = BlattWeisskopf{1}(3.0)
        root_mdep = root_ff(tfpwa_breakup(ctx.m_B_root, mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(tfpwa_breakup(nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]))
        amp_l0 = root_mdep * eval_chain(ctx, l0, 2, (2, 2), (0, 2))
        amp_l2 = root_mdep * eval_chain(ctx, l2, 2, (2, 2), (4, 2))
        total_key = name == "X(3940)(1.)" ? "Bp->X(3940)(1.).KX(3940)(1.)->Dst.DDst->D0.pi_total_0" : "Bp->$(name).K$(name)->Dst.DDst->D0.pi_total_0"
        return param_complex(total_key) * (amp_l0 + param_complex("$(name)->Dst.D_g_ls_1") * amp_l2)
    elseif name == "Psi(4040)"
        raw = eval_chain(ctx, bwr_lineshape(ctx, nominal_mass[name], param_real(name * "_width"), 1, 1.0 + 0im), 2, (2, 2), (2, 2); root_l = 1, decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]) * mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(0-)SPp"
        alpha = param_real("NR(0-)SPp_alpha"); beta = param_real("NR(0-)SPp_beta")
        nr_factor = -exp(-(alpha + 1im * beta) * (mass(ctx.P_R)^2 - nominal_mass["NR(0-)SPp"]^2))
        raw = eval_chain(ctx, ConstantLineshape(nr_factor), 0, (0, 0), (2, 2); decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["NR(0-)SPp"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(0-)SPp.KNR(0-)SPp->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(1.)PSp"
        raw = eval_chain(ctx, ConstantLineshape(-1.0 + 0im), 2, (2, 2), (0, 2); root_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["NR(1.)PSp"], nominal_mass["K"])
        return param_complex("Bp->NR(1.)PSp.KNR(1.)PSp->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(0-)SPm"
        raw = eval_chain(ctx, ConstantLineshape(1.0 + 0im), 0, (0, 0), (2, 2); decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["NR(0-)SPm"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(0-)SPm.KNR(0-)SPm->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "NR(1-)PPm"
        raw = eval_chain(ctx, ConstantLineshape(1.0 + 0im), 2, (2, 2), (2, 2); root_l = 1, decay_l = 1)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass["NR(1-)PPm"], nominal_mass["K"]) * mismatch_factor(1, 3.0, nominal_mass["NR(1-)PPm"], nominal_mass["Dst"], nominal_mass["D"])
        return param_complex("Bp->NR(1-)PPm.KNR(1-)PPm->Dst.DDst->D0.pi_total_0") * raw * correction
    elseif name == "X0(2900)"
        raw = eval_dk_chain(ctx, x2900_bwr_lineshape(ctx, name, 0, param_real), 0, (2, 2), (0, 0); root_l = 1, dk_l = 0)
        correction = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) * mismatch_factor(0, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        return param_complex("Bp->X0(2900).DstX0(2900)->D.KDst->D0.pi_total_0") * raw * correction
    elseif name == "X1(2900)"
        lineshape = x2900_bwr_lineshape(ctx, name, 1, param_real)
        raw_l0 = eval_dk_chain(ctx, lineshape, 2, (0, 0), (2, 0); root_l = 0, dk_l = 1, remove_root_particle2_phase = true)
        raw_l1 = eval_dk_chain(ctx, lineshape, 2, (2, 2), (2, 0); root_l = 1, dk_l = 1, remove_root_particle2_phase = true)
        raw_l2 = eval_dk_chain(ctx, lineshape, 2, (4, 4), (2, 0); root_l = 2, dk_l = 1, remove_root_particle2_phase = true)
        correction_l0 = mismatch_factor(0, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) * mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        correction_l1 = mismatch_factor(1, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) * mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        correction_l2 = mismatch_factor(2, 3.0, nominal_mass["Bp"], nominal_mass[name], nominal_mass["Dst"]) * mismatch_factor(1, 3.0, nominal_mass[name], nominal_mass["D"], nominal_mass["K"])
        coherent = raw_l0 * correction_l0 + param_complex("Bp->X1(2900).Dst_g_ls_1") * raw_l1 * correction_l1 + param_complex("Bp->X1(2900).Dst_g_ls_2") * raw_l2 * correction_l2
        return param_complex("Bp->X1(2900).DstX1(2900)->D.KDst->D0.pi_total_0") * coherent
    end
    error("Unknown resonance: $(name)")
end

# Main execution
function main()
    work_dir = @__DIR__
    suite_dir = normpath(joinpath(work_dir, ".."))
    root_candidates = [
        normpath(joinpath(suite_dir, "..")),
        normpath(joinpath(suite_dir, "..", "B2DxDK.jl")),
        "c:/Users/gamma/Documents/Playground/Antigravity_Test",
        "c:/Users/gamma/Documents/Playground/Antigravity_Test/B2DxDK.jl",
    ]

    events_path = if length(ARGS) >= 1
        ARGS[1]
    else
        found = nothing
        for r in root_candidates
            cand = joinpath(r, "data", "sampled_events_tfpwa.json")
            if isfile(cand)
                found = cand; break
            end
        end
        found !== nothing ? found : joinpath(suite_dir, "..", "data", "sampled_events_tfpwa.json")
    end

    output_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(suite_dir, "amp_data", "case_b_cd_amp.txt")

    params_path = nothing
    for r in root_candidates
        for sub in ["data", "Analysis", "archive/investigation/Analysis"]
            cand = joinpath(r, sub, "final_params_full.json")
            if isfile(cand)
                params_path = cand; break
            end
        end
        params_path !== nothing && break
    end
    params_path === nothing && error("Could not find final_params_full.json")

    params = JSON.parsefile(params_path)["value"]
    for name in ["X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)", "X(3993)", "Psi(4040)", "X(4300)", "X0(2900)", "X1(2900)"]
        haskey(params, name * "_mass") && (nominal_mass[name] = Float64(params[name * "_mass"]))
    end
    param_real(key) = Float64(params[key])
    param_complex(key) = param_real(key * "r") * cis(param_real(key * "i"))

    println("Case (b): PDG Parent Masses + Nominal Lineshapes")
    println("Loading sampled events from: ", events_path)
    data = JSON.parsefile(events_path)
    p4_data = data["p4"]
    n_events = length(p4_data["D"])
    println("Loaded $(n_events) events.")

    p4_D  = [Float64.(p4_data["D"][i])  for i in 1:n_events]
    p4_K  = [Float64.(p4_data["K"][i])  for i in 1:n_events]
    p4_D0 = [Float64.(p4_data["D0"][i]) for i in 1:n_events]
    p4_pi = [Float64.(p4_data["pi"][i]) for i in 1:n_events]
    amps  = Vector{ComplexF64}(undef, n_events)

    println("Evaluating coherent amplitude on $(Threads.nthreads()) Julia threads...")
    t0 = time()
    Threads.@threads for idx in 1:n_events
        evt = Dict("D" => p4_D[idx], "K" => p4_K[idx], "D0" => p4_D0[idx], "pi" => p4_pi[idx])
        ctx = build_event_context(evt)
        amps[idx] = sum(selected_cd_amplitude(ctx, name, param_real, param_complex) for name in all_resonance_names)
    end
    elapsed = time() - t0
    @printf("Evaluation completed in %.2f seconds (%.1f events/s across %d threads).\n", elapsed, n_events / elapsed, Threads.nthreads())

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, "# real imag")
        for a in amps
            @printf(io, "%.17e %.17e\n", real(a), imag(a))
        end
    end
    println("Saved complex amplitudes (ASCII text) to: ", output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
