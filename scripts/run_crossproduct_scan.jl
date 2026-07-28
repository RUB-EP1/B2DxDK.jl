#!/usr/bin/env julia
#
# Multi-point version of `run_crossproduct_walk.jl`.
#
# `run_crossproduct_walk.jl` establishes on ONE event that the cross-product
# construction and the realigning `ToHelicityFrame` construction give the same
# vertex angles.  One event can always be a coincidence, so this scans N random
# phase-space points and reports the worst disagreement over all of them.
#
# Scope, deliberately narrow — this tests VERTEX ANGLES ONLY:
#   • both topologies, all three vertices each (6 angles per event)
#   • events generated in the B rest frame, compared with `CurrentFrame()`
#   • a random rotation per event, so the (ẑ, x̂) transport is exercised at
#     arbitrary orientations rather than the one in crosscheck_event.json
#
# It does NOT test the external Wigner alignment path (`helicity_frame_path`,
# `wigner_finals`), which is where `ToHelicityFrameParticle2` actually appears
# and where a genuine difference could still live.  It says nothing about
# amplitudes.  See the design note for why angle agreement is expected.
#
# Run from the repo root (see scripts/README.md for the JULIA_LOAD_PATH stanza):
#   JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/run_crossproduct_scan.jl [N]

using B2DxDK
using CascadeDecays
using FourVectors
using InstructionalDecayTrees
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "CrossProductWalk.jl"))
using .CrossProductWalk

const AX = 5
const TOL = 1.0e-10

wrap_delta(a, b) = mod(a - b + π, 2π) - π

# ---------------------------------------------------------------------------
# Phase-space generation, in the B rest frame
# ---------------------------------------------------------------------------

"""Inverse of `boost_to_rest`: carry `q` from P's rest frame into P's frame."""
boost_from_rest(q, P) = boost_to_rest(q, FourVector(-P.px, -P.py, -P.pz; E = P.E))

"""Isotropic two-body decay of a mass-`M` parent at rest."""
function two_body(rng, M, m1, m2)
    p = sqrt((M^2 - (m1 + m2)^2) * (M^2 - (m1 - m2)^2)) / (2M)
    cosθ = 2rand(rng) - 1
    sinθ = sqrt(1 - cosθ^2)
    ϕ = 2π * rand(rng)
    v = (p * sinθ * cos(ϕ), p * sinθ * sin(ϕ), p * cosθ)
    return (
        FourVector(v...; E = sqrt(p^2 + m1^2)),
        FourVector((-).(v)...; E = sqrt(p^2 + m2^2)),
    )
end

"""
One B → D⁰ π D K configuration in the B rest frame, then randomly rotated.

Generated through the DxD chain, but the comparison runs both topologies, so
the generation route biases neither side.
"""
function random_event(rng, m)
    mDst = m.D0 + m.pi + rand(rng) * (m.B - m.K - m.D - m.D0 - m.pi)
    mX = mDst + m.D + rand(rng) * (m.B - m.K - mDst - m.D)

    X, K = two_body(rng, m.B, mX, m.K)
    Dst_x, D_x = two_body(rng, mX, mDst, m.D)
    Dst, D = boost_from_rest(Dst_x, X), boost_from_rest(D_x, X)
    D0_d, pi_d = two_body(rng, mDst, m.D0, m.pi)
    D0, pip = boost_from_rest(D0_d, Dst), boost_from_rest(pi_d, Dst)

    # One rotation for the whole event: draw the angles ONCE, outside the closure.
    a, b, c = 2π * rand(rng), acos(2rand(rng) - 1), 2π * rand(rng)
    R = p -> p |> Rz(a) |> Ry(b) |> Rz(c)
    return map(R, (D0, pip, D, K))
end

# ---------------------------------------------------------------------------
# The six vertices: walk program + CascadeDecays address
# ---------------------------------------------------------------------------

const VERTICES = [
    ("DxD", "B → X K", B2DxDK.dxd_topology, (((1, 2), 3), 4), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        MeasureEulerZXZ(:ang, (1, 2, 3), AX),
    )),
    ("DxD", "X → D* D", B2DxDK.dxd_topology, ((1, 2), 3), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2, 3)), ToRestFrame((1, 2, 3)),
        MeasureEulerZXZ(:ang, (1, 2), AX),
    )),
    ("DxD", "D* → D⁰ π", B2DxDK.dxd_topology, (1, 2), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2, 3)), ToRestFrame((1, 2, 3)),
        TransportAxes(AX, (1, 2)), ToRestFrame((1, 2)),
        MeasureEulerZXZ(:ang, (1,), AX),
    )),
    ("dk", "B → D* X", B2DxDK.dk_topology, ((1, 2), (3, 4)), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        MeasureEulerZXZ(:ang, (1, 2), AX),
    )),
    ("dk", "D* → D⁰ π", B2DxDK.dk_topology, (1, 2), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2)), ToRestFrame((1, 2)),
        MeasureEulerZXZ(:ang, (1,), AX),
    )),
    # second daughter of the root — the child-2 branch
    ("dk", "X → D K", B2DxDK.dk_topology, (3, 4), (
        PlantLabAxes(AX), ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (3, 4)), ToRestFrame((3, 4)),
        MeasureEulerZXZ(:ang, (3,), AX),
    )),
]

function main(N::Int)
    rng = MersenneTwister(20260728)
    m = (
        B = B2DxDK.nominal_mass["Bp"], D0 = B2DxDK.nominal_mass["D0"],
        pi = B2DxDK.nominal_mass["pi"], D = B2DxDK.nominal_mass["D"],
        K = B2DxDK.nominal_mass["K"],
    )

    println("=" ^ 74)
    println("CROSS-PRODUCT WALK vs CascadeDecays — scan over $N phase-space points")
    println("=" ^ 74)
    println("Vertex angles only.  Events in the B rest frame, CurrentFrame(),")
    println("random orientation per event.  Alignment path NOT covered.\n")

    worst_α = zeros(length(VERTICES))
    worst_β = zeros(length(VERTICES))
    for _ in 1:N
        objs = random_event(rng, m)
        for (k, (_, _, topo, address, program)) in enumerate(VERTICES)
            (_, res) = apply_decay_instruction(program, with_axes(objs))
            cd = vertex_angles(
                CascadeKinematics(topo, objs; initial_frame = CurrentFrame()), topo, address)
            worst_α[k] = max(worst_α[k], abs(wrap_delta(res.ang.α, cd.ϕ)))
            worst_β[k] = max(worst_β[k], abs(res.ang.β - acos(clamp(cd.cosθ, -1, 1))))
        end
    end

    @printf("  %-4s %-12s %14s %14s\n", "topo", "vertex", "max |α−ϕ|", "max |β−θ|")
    for (k, (t, label, _, _, _)) in enumerate(VERTICES)
        @printf("  %-4s %-12s %14.3e %14.3e\n", t, label, worst_α[k], worst_β[k])
    end
    overall = max(maximum(worst_α), maximum(worst_β))
    @printf("\n  worst disagreement over all %d events x 6 vertices: %.3e rad\n", N, overall)

    ok = overall < TOL
    println(ok ? "\nPASS — agreement is not a coincidence of one event." :
        "\nFAIL — disagreement exceeds $TOL rad.")
    return ok
end

main(isempty(ARGS) ? 1000 : parse(Int, ARGS[1])) || exit(1)
