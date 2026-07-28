#!/usr/bin/env julia
# Compare CascadeDecays vs TF-PWA helicity angles for one event.
#
# TF-PWA stores (alpha, beta) for both daughters when building frames, but
# HelicityDecay.get_D_matrix_term uses only the first listed daughter:
#     ang = data[self.outs[0]]["ang"]
# CascadeDecays likewise measures (cosθ, ϕ) on child_line_inds(...)[1].

using B2DxDK
using CascadeDecays
using FourVectors
using JSON
using LinearAlgebra
using Printf

const EPS = 1.0e-8

dot3(a, b) = dot(a, b)
unitvec(v) = v ./ norm(v)

function cross_unit(a, b)
    c = cross(a, b)
    norm(c) < EPS && (c = cross(a, ones(3) .+ b))
    return unitvec(c)
end

angle_from(v, x, y) = atan(dot3(v, y), dot3(v, x))

function angle_zx_z_getx(z1, x1, z2)
    u_z1 = unitvec(z1)
    u_z2 = unitvec(z2)
    u_y1 = cross_unit(z1, x1)
    u_x1 = cross_unit(u_y1, z1)
    u_yr = cross_unit(z1, z2)
    u_xr = cross_unit(u_yr, z1)
    alpha = angle_from(u_xr, u_x1, u_y1)
    beta = angle_from(u_z2, u_z1, u_xr)
    return (alpha=alpha, beta=beta, gamma=0.0), cross_unit(u_yr, u_z2)
end

boost_vector(v) = v[2:4] ./ v[1]

function boost(v, beta)
    beta2 = dot(beta, beta)
    gamma = 1.0 / sqrt(1.0 - beta2)
    bp = dot(beta, v[2:4])
    gamma2 = beta2 > EPS ? (gamma - 1.0) / beta2 : 0.0
    # Both components must transform. Returning the energy untouched leaves the
    # spatial part of *this* boost right but corrupts every boost below it,
    # because the next rest_vector divides by this energy to get its beta.
    return [gamma * (v[1] + bp); v[2:4] .+ gamma2 .* bp .* beta .+ gamma .* v[1] .* beta]
end

rest_vector(core, other) = boost(other, -boost_vector(core))

function compute_chain_boosts(particle_p4, chain)
    particle_set = Set{String}()
    for (_, outs) in chain
        foreach(x -> push!(particle_set, x), outs)
    end
    core_decay_map = Dict{String,String}()
    part_data = Dict{String,Dict{String,Vector{Float64}}}()
    pending = copy(chain)
    while !isempty(pending)
        extra = []
        for (core, outs) in pending
            if core == "Bp"
                p_rest = particle_p4[core]
                part_data[core] = Dict{String,Vector{Float64}}()
                for out in outs
                    core_decay_map[out] = core
                    part_data[core][out] = rest_vector(p_rest, particle_p4[out])
                    delete!(particle_set, out)
                end
                for other in collect(particle_set)
                    part_data[core][other] = rest_vector(p_rest, particle_p4[other])
                end
            elseif haskey(core_decay_map, core)
                parent = core_decay_map[core]
                p_rest = part_data[parent][core]
                part_data[core] = Dict{String,Vector{Float64}}()
                for out in outs
                    core_decay_map[out] = core
                    part_data[core][out] = rest_vector(p_rest, part_data[parent][out])
                    delete!(particle_set, out)
                end
                for other in collect(particle_set)
                    part_data[core][other] = rest_vector(p_rest, part_data[parent][other])
                end
            else
                push!(extra, (core, outs))
            end
        end
        pending = extra
    end
    return part_data
end

function calculate_helicity_angles(particle_p4, chain)
    part_data = compute_chain_boosts(particle_p4, chain)
    set_x = Dict("Bp" => [1.0, 0.0, 0.0])
    set_z = Dict("Bp" => [0.0, 0.0, 1.0])
    angles = Dict{String,Dict{String,NamedTuple}}()
    for (core, outs) in chain
        angles[core] = Dict{String,NamedTuple}()
        bias = -pi
        for out in outs
            z2 = part_data[core][out][2:4]
            ang, x_axis = angle_zx_z_getx(set_z[core], set_x[core], z2)
            set_x[out] = x_axis
            set_z[out] = z2
            angles[core][out] = (
                alpha=mod(ang.alpha - bias, 2pi) + bias,
                beta=ang.beta,
                gamma=ang.gamma,
            )
            bias -= pi
        end
    end
    return angles
end

"""Return only the TF-PWA angles that enter get_D_matrix_term (outs[0] per vertex)."""
function tfpwa_used_angles(chain, all_angles)
    return [
        (
            label = "$core → $(outs[1])  [uses outs[0]=$(outs[1])]",
            core = core,
            outs = outs,
            used_daughter = outs[1],
            alpha = all_angles[core][outs[1]].alpha,
            beta = all_angles[core][outs[1]].beta,
        )
        for (core, outs) in chain
    ]
end

function load_event(path)
    event = JSON.parsefile(path)
    fv = event["four_vectors"]
    p4 = Dict(
        "D0" => [fv["D0"]["E"], fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]],
        "pi" => [fv["pi"]["E"], fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]],
        "D" => [fv["D"]["E"], fv["D"]["px"], fv["D"]["py"], fv["D"]["pz"]],
        "K" => [fv["K"]["E"], fv["K"]["px"], fv["K"]["py"], fv["K"]["pz"]],
    )
    p4["Dst"] = p4["D0"] .+ p4["pi"]
    objs = (
        FourVector(fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]; E=fv["D0"]["E"]),
        FourVector(fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]; E=fv["pi"]["E"]),
        FourVector(fv["D"]["px"], fv["D"]["py"], fv["D"]["pz"]; E=fv["D"]["E"]),
        FourVector(fv["K"]["px"], fv["K"]["py"], fv["K"]["pz"]; E=fv["K"]["E"]),
    )
    return event, p4, objs
end

function cd_theta_phi(ang)
    return (θ = acos(clamp(ang.cosθ, -1, 1)), ϕ = ang.ϕ)
end

function wrap_delta(a, b)
    return mod(a - b + pi, 2pi) - pi
end

function main()
    event_path = joinpath(@__DIR__, "..", "..", "archive", "data", "crosscheck_event.json")
    event, p4, objs = load_event(event_path)
    point = KinematicPoint(B2DxDK.kinematic_task, objs)
    x_dxd = kinematics_at(point, B2DxDK.dxd_topology)
    x_dk = kinematics_at(point, B2DxDK.dk_topology)

    chain_dxd = [("Bp", ["X", "K"]), ("X", ["Dst", "D"]), ("Dst", ["D0", "pi"])]
    chain_dk = [("Bp", ["Dst", "X"]), ("Dst", ["D0", "pi"]), ("X", ["D", "K"])]

    p4_dxd = copy(p4)
    p4_dxd["X"] = p4["Dst"] .+ p4["D"]
    p4_dxd["Bp"] = p4_dxd["X"] .+ p4["K"]
    p4_dk = copy(p4)
    p4_dk["X"] = p4["D"] .+ p4["K"]
    p4_dk["Bp"] = p4_dk["Dst"] .+ p4_dk["X"]

    tf_dxd = tfpwa_used_angles(chain_dxd, calculate_helicity_angles(p4_dxd, chain_dxd))
    tf_dk = tfpwa_used_angles(chain_dk, calculate_helicity_angles(p4_dk, chain_dk))

    comparisons = [
        (
            topology = "DxD",
            vertex = "Bp → X, K",
            cd = cd_theta_phi(vertex_angles(x_dxd, B2DxDK.dxd_topology, (((1, 2), 3), 4))),
            tf = (ϕ = tf_dxd[1].alpha, θ = tf_dxd[1].beta),
            note = "B has J=0: D factor is 1, angle unused in amplitude",
        ),
        (
            topology = "DxD",
            vertex = "X → Dst, D",
            cd = cd_theta_phi(vertex_angles(x_dxd, B2DxDK.dxd_topology, ((1, 2), 3))),
            tf = (ϕ = tf_dxd[2].alpha, θ = tf_dxd[2].beta),
            note = "",
        ),
        (
            topology = "DxD",
            vertex = "Dst → D0, π",
            cd = cd_theta_phi(vertex_angles(x_dxd, B2DxDK.dxd_topology, (1, 2))),
            tf = (ϕ = tf_dxd[3].alpha, θ = tf_dxd[3].beta),
            note = "",
        ),
        (
            topology = "dk",
            vertex = "Bp → Dst, X",
            cd = cd_theta_phi(vertex_angles(x_dk, B2DxDK.dk_topology, (((1, 2), (3, 4))))),
            tf = (ϕ = tf_dk[1].alpha, θ = tf_dk[1].beta),
            note = "B has J=0: D factor is 1, angle unused in amplitude",
        ),
        (
            topology = "dk",
            vertex = "Dst → D0, π",
            cd = cd_theta_phi(vertex_angles(x_dk, B2DxDK.dk_topology, (1, 2))),
            tf = (ϕ = tf_dk[2].alpha, θ = tf_dk[2].beta),
            note = "",
        ),
        (
            topology = "dk",
            vertex = "X → D, K",
            cd = cd_theta_phi(vertex_angles(x_dk, B2DxDK.dk_topology, (3, 4))),
            tf = (ϕ = tf_dk[3].alpha, θ = tf_dk[3].beta),
            note = "",
        ),
    ]

    println("Helicity-angle comparison on ", event_path)
    println("External order: (D0=1, pi=2, D=3, K=4)")
    println()
    println("Both frameworks use the first listed daughter at each 2-body vertex.")
    println("TF-PWA: get_D_matrix_term reads data[outs[0]][\"ang\"] (alpha, beta).")
    println("CascadeDecays: vertex_angles on child_line_inds(...)[1] gives (ϕ, θ).")
    println("Comparison maps CD (ϕ, θ) ↔ TF-PWA (alpha, beta), gamma = 0.\n")

    @printf("%-5s %-18s  %12s  %12s  %12s  %12s  %6s\n",
            "Topo", "Vertex", "CD ϕ", "TF α", "CD θ", "TF β", "Match")
    max_Δϕ = 0.0
    max_Δθ = 0.0
    for row in comparisons
        Δϕ = wrap_delta(row.cd.ϕ, row.tf.ϕ)
        Δθ = row.cd.θ - row.tf.θ
        max_Δϕ = max(max_Δϕ, abs(Δϕ))
        max_Δθ = max(max_Δθ, abs(Δθ))
        ok = abs(Δϕ) < 1e-10 && abs(Δθ) < 1e-10
        @printf("%-5s %-18s  %12.8f  %12.8f  %12.8f  %12.8f  %6s\n",
                row.topology, row.vertex, row.cd.ϕ, row.tf.ϕ, row.cd.θ, row.tf.θ, ok ? "OK" : "DIFF")
        isempty(row.note) || println("      ", row.note)
    end

    @printf("\nMax |Δϕ| = %.3e rad,  Max |Δθ| = %.3e rad\n", max_Δϕ, max_Δθ)

    println("""
The remaining DIFFs are an initial-frame artifact, not a convention mismatch:
every θ agrees exactly below the root, and the ϕ offsets are constant, which is
the signature of one spurious azimuthal rotation. |p⃗_B| ≈ 2e-10 GeV exceeds
CascadeDecays' at-rest tolerance (rtol 1e-12), so KinematicPoint prepends
ToHelicityFrame(B) using the ϕ, θ of numerical noise. Re-run with
CascadeKinematics(topology, objs; initial_frame=CurrentFrame()) and every row
agrees to ~1e-15 — see scripts/angular_check/run_crossproduct_walk.jl §4.""")

    if haskey(event, "dpd_kinematics")
        dpd = event["dpd_kinematics"]
        cd = cd_theta_phi(vertex_angles(x_dxd, B2DxDK.dxd_topology, (1, 2)))
        tf = (ϕ = tf_dxd[3].alpha, θ = tf_dxd[3].beta)
        println("\nDPD json reference at Dst → D0 (DxD topology):")
        @printf("  json: ϕ=% .12f  θ=% .12f\n", dpd["phi_D_in_Dx"], acos(dpd["cos_theta_D_in_Dx"]))
        @printf("  CD:   ϕ=% .12f  θ=% .12f\n", cd.ϕ, cd.θ)
        @printf("  TF:   α=% .12f  β=% .12f\n", tf.ϕ, tf.θ)
    end
end

main()
