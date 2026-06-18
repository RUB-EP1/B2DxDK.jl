using Arrow
using CascadeDecays
using DataFrames
using FourVectors
using HadronicLineshapes
using JSON
using Printf
using Statistics
import ThreeBodyDecays
using ThreeBodyDecays: Recoupling, RecouplingLS, VertexFunction

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

const repo_root = normpath(joinpath(@__DIR__, ".."))
const data_path = joinpath(repo_root, "data", "b-decay-events.arrow")
const params_path = joinpath(repo_root, "Analysis", "final_params_full.json")
const output_path = joinpath(repo_root, "scripts", "all_resonances_fit_fractions.txt")
const saved_reference_path = joinpath(repo_root, "notebooks", "all_resonances_fit_fractions.txt")

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

const all_resonance_names = [
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
    "X0(2900)",
    "X1(2900)",
]

const topology = DecayTopology((((1, 2), 3), 4))
const dk_topology = DecayTopology(((1, 2), (3, 4)))

params = JSON.parsefile(params_path)["value"]
for name in ["X(3872)", "X(3915)(0-)", "chi(c2)(3930)", "X(3940)(1.)", "X(3993)", "Psi(4040)", "X(4300)", "X0(2900)", "X1(2900)"]
    haskey(params, name * "_mass") && (nominal_mass[name] = Float64(params[name * "_mass"]))
end

param_real(key) = Float64(params[key])
param_complex(key) = param_real(key * "r") * cis(param_real(key * "i"))

breakup_momentum(m0, m1, m2) =
    sqrt(complex(((m0^2 - (m1 + m2)^2) * (m0^2 - (m1 - m2)^2)) / (4.0 * m0^2)))

mismatch_factor(l, d, m0, m1, m2) = 1 / BlattWeisskopf{l}(d)(m0^2, m1^2, m2^2)

function row_to_sampled_p4(row)
    return Dict(
        "D" => [row.Dm_E, row.Dm_px, row.Dm_py, row.Dm_pz],
        "K" => [row.Kp_E, row.Kp_px, row.Kp_py, row.Kp_pz],
        "D0" => [row.D0_E, row.D0_px, row.D0_py, row.D0_pz],
        "pi" => [row.pip_E, row.pip_px, row.pip_py, row.pip_pz],
    )
end

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

bwr_lineshape(ctx, m0, width, l, sign) = begin
    q0 = real(breakup_momentum(m0, nominal_mass["Dst"], nominal_mass["D"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = m0 * width / (2q0) * m0 / ff(q0)^2
    sign * MultichannelBreitWigner(m0, [(; gsq, ma = nominal_mass["Dst"], mb = nominal_mass["D"], l, d = 3.0)])
end

function ad_hoc_mass(m0, m_min, m_max)
    k = (m_max - m_min) / 2
    return k * (1 + tanh((2m0 - (m_max + m_min)) / k / 4)) + m_min
end

function bwr_ls_lineshapes(ctx, name, sign; below_threshold = false)
    m0 = nominal_mass[name]
    q0_mass = below_threshold ? ad_hoc_mass(m0, nominal_mass["Dst"] + nominal_mass["D"], nominal_mass["Bp"] - nominal_mass["K"]) : m0
    q0 = real(breakup_momentum(q0_mass, nominal_mass["Dst"], nominal_mass["D"]))
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
    breakup_from_sigma = sigma -> breakup_momentum(sqrt(sigma), nominal_mass["Dst"], nominal_mass["D"])
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

function dk_chain_amplitude(ctx, lineshape, two_j, root_two_ls, dk_two_ls; root_l = nothing, dk_l = nothing, remove_root_particle2_phase = false)
    root_recoupling = remove_root_particle2_phase ? RemoveParticleTwoPhaseLS(root_two_ls) : RecouplingLS(root_two_ls)
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
    q0 = real(breakup_momentum(nominal_mass[name], nominal_mass["D"], nominal_mass["K"]))
    ff = BlattWeisskopf{l}(3.0)
    gsq = nominal_mass[name] * param_real(name * "_width") / (2q0) * nominal_mass[name] / ff(q0)^2
    return MultichannelBreitWigner(
        nominal_mass[name],
        [(; gsq, ma = nominal_mass["D"], mb = nominal_mass["K"], l, d = 3.0)],
    )
end

function selected_cd_amplitude(ctx, name::String)
    if name == "X(3872)"
        l0, l2 = bwr_ls_lineshapes(ctx, "X(3872)", -1.0 + 0im; below_threshold = true)
        root_ff = BlattWeisskopf{1}(3.0)
        root_mdep = root_ff(breakup_momentum(mass(ctx.P_B), mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(breakup_momentum(nominal_mass["Bp"], nominal_mass["X(3872)"], nominal_mass["K"]))
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
        root_mdep = root_ff(breakup_momentum(mass(ctx.P_B), mass(ctx.P_R), mass(ctx.pKplus))) /
                    root_ff(breakup_momentum(nominal_mass["Bp"], nominal_mass[name], nominal_mass["K"]))
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
        raw_l0 = dk_chain_amplitude(dk_ctx, lineshape, 2, (0, 0), (2, 0); root_l = 0, dk_l = 1, remove_root_particle2_phase = true)
        raw_l1 = dk_chain_amplitude(dk_ctx, lineshape, 2, (2, 2), (2, 0); root_l = 1, dk_l = 1, remove_root_particle2_phase = true)
        raw_l2 = dk_chain_amplitude(dk_ctx, lineshape, 2, (4, 4), (2, 0); root_l = 2, dk_l = 1, remove_root_particle2_phase = true)
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
    saved = Dict{String, Float64}()
    for line in eachline(path)
        startswith(line, "component") && continue
        fields = split(line, '\t')
        length(fields) >= 3 || continue
        saved[fields[1]] = parse(Float64, fields[3])
    end
    return saved
end

function main()
    println("All-resonance fit fractions from b-decay-events.arrow")
    println("====================================================")
    println("Input:  ", data_path)
    println("Output: ", output_path)
    println()

    df = DataFrame(Arrow.Table(data_path))
    n_events = nrow(df)
    weights = df.weight
    println("Loaded events: ", n_events)
    println("Using event weights from column: weight")
    println(@sprintf("  sum(weight) = %.6e", sum(weights)))
    println("Computing per-resonance amplitudes...")

    resonance_amps_by_event = Vector{Vector{ComplexF64}}()
    sizehint!(resonance_amps_by_event, n_events)

    for idx in 1:n_events
        ctx = event_context(row_to_sampled_p4(df[idx, :]))
        push!(resonance_amps_by_event, ComplexF64[selected_cd_amplitude(ctx, name) for name in all_resonance_names])
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
        return fit_fractions, interference_fraction
    end

    println()
    println("Comparison to saved reference (", basename(saved_reference_path), "):")
    println(@sprintf("  %-16s %12s %12s %12s", "component", "computed", "saved", "delta"))
    max_abs_delta = 0.0
    for name in all_resonance_names
        saved_val = get(saved, name, NaN)
        delta = fit_fractions[findfirst(==(name), all_resonance_names)] - saved_val
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", name, fit_fractions[findfirst(==(name), all_resonance_names)], saved_val, delta))
    end
    if haskey(saved, "interference")
        delta = interference_fraction - saved["interference"]
        max_abs_delta = max(max_abs_delta, abs(delta))
        println(@sprintf("  %-16s %12.6e %12.6e %12.6e", "interference", interference_fraction, saved["interference"], delta))
    end
    println(@sprintf("Max |delta| = %.6e", max_abs_delta))
    return fit_fractions, interference_fraction
end

main()
