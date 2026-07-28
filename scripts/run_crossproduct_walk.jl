#!/usr/bin/env julia
#
# Validation for `CrossProductWalk.jl` — TF-PWA cross-product helicity angles
# expressed as an InstructionalDecayTrees program.
#
# Run from the repo root (see scripts/README.md for the JULIA_LOAD_PATH stanza):
#
#   TENV=$(mktemp -d); cp Manifest.toml "$TENV/"
#   printf '[deps]\nInstructionalDecayTrees = "1d606af4-d0f8-4ff7-bc8d-eb3f657b7647"\n' > "$TENV/Project.toml"
#   JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/run_crossproduct_walk.jl
#
# Checks, in order:
#   1. frame trace          — what frame `objs` are in after every program line
#   2. TF-PWA equivalence   — walk ≡ standalone port of cal_chain_boost + angle_zx_z_getx
#   3. IDT equivalence      — walk ≡ ToHelicityFrame chain + MeasureCosThetaPhi
#   4. CascadeDecays        — walk vs CurrentFrame / HelicityRootFrame / KinematicPoint
#   5. frame algebra        — ToHelicityFrame == ToRestFrame followed by the realignment

using B2DxDK
using CascadeDecays
using FourVectors
using InstructionalDecayTrees
using JSON
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "CrossProductWalk.jl"))
include(joinpath(@__DIR__, "TFPWACrossProductHelicity.jl"))

using .CrossProductWalk
using .TFPWACrossProductHelicity: run_crossproduct_chain

const TOL_MACHINE = 1.0e-8   # walk vs independent implementations
const TOL_ROOT = 1.0e-8      # brief's criterion at the root vertex

wrap_delta(a, b) = mod(a - b + π, 2π) - π
p3(p::FourVector) = [p.px, p.py, p.pz]
mom(objs, idx) = sum(objs[i] for i in idx)

# ---------------------------------------------------------------------------
# Event.  External order is (D0 = 1, π = 2, D = 3, K = 4) throughout.
# ---------------------------------------------------------------------------

function load_event()
    event = JSON.parsefile(joinpath(@__DIR__, "..", "archive", "data", "crosscheck_event.json"))
    fv = event["four_vectors"]
    objs = (
        FourVector(fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]; E = fv["D0"]["E"]),
        FourVector(fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]; E = fv["pi"]["E"]),
        FourVector(fv["D"]["px"], fv["D"]["py"], fv["D"]["pz"]; E = fv["D"]["E"]),
        FourVector(fv["K"]["px"], fv["K"]["py"], fv["K"]["pz"]; E = fv["K"]["E"]),
    )
    lab = Dict(
        :D0 => [fv["D0"]["E"], fv["D0"]["px"], fv["D0"]["py"], fv["D0"]["pz"]],
        :pi => [fv["pi"]["E"], fv["pi"]["px"], fv["pi"]["py"], fv["pi"]["pz"]],
        :D => [fv["D"]["E"], fv["D"]["px"], fv["D"]["py"], fv["D"]["pz"]],
        :K => [fv["K"]["E"], fv["K"]["px"], fv["K"]["py"], fv["K"]["pz"]],
    )
    lab[:Dst] = lab[:D0] .+ lab[:pi]
    lab[:X_dxd] = lab[:Dst] .+ lab[:D]
    lab[:X_dk] = lab[:D] .+ lab[:K]
    lab[:Bp] = lab[:X_dxd] .+ lab[:K]
    return event, objs, lab
end

# ---------------------------------------------------------------------------
# The programs.  One per vertex, each a straight-line walk from the root —
# the same idiom CascadeDecays uses (`helicity_angle_program` restarts from the
# root for every vertex), which is why no backtracking instruction is needed.
#
# Slot 5 is the helicity-axis marker.  It is reused at every depth: a
# straight-line walk only ever needs the axes of the core it is sitting in.
# ---------------------------------------------------------------------------

const AX = 5

const WALK_PROGRAMS = [
    # topology, depth, label, address for CascadeDecays, program
    ("DxD", 1, "B → X K   (measure X)", (((1, 2), 3), 4), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        MeasureEulerZXZ(:ang, (1, 2, 3), AX),
    )),
    ("DxD", 2, "X → D* D  (measure D*)", ((1, 2), 3), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2, 3)),
        ToRestFrame((1, 2, 3)),
        MeasureEulerZXZ(:ang, (1, 2), AX),
    )),
    ("DxD", 3, "D* → D⁰ π  (measure D⁰)", (1, 2), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2, 3)),
        ToRestFrame((1, 2, 3)),
        TransportAxes(AX, (1, 2)),
        ToRestFrame((1, 2)),
        MeasureEulerZXZ(:ang, (1,), AX),
    )),
    ("dk", 1, "B → D* X  (measure D*)", ((1, 2), (3, 4)), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        MeasureEulerZXZ(:ang, (1, 2), AX),
    )),
    ("dk", 2, "D* → D⁰ π  (measure D⁰)", (1, 2), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (1, 2)),
        ToRestFrame((1, 2)),
        MeasureEulerZXZ(:ang, (1,), AX),
    )),
    # Second daughter of the root: the walk descends into X = (D, K).  Same
    # TransportAxes instruction, no bias — the branch convention only ever
    # touches a *measured* azimuth, never a transported axis.
    ("dk", 3, "X → D K   (measure D)", (3, 4), (
        PlantLabAxes(AX),
        ToRestFrame((1, 2, 3, 4)),
        TransportAxes(AX, (3, 4)),
        ToRestFrame((3, 4)),
        MeasureEulerZXZ(:ang, (3,), AX),
    )),
]

"""
The plain-IDT program measuring the same vertex the realigning way.

Note what is NOT here: a leading `ToHelicityFrame((1,2,3,4))`.  The event is
already in the B rest frame, and on a 2e-10 GeV momentum that instruction would
rotate the whole event by the ϕ, θ of numerical noise.  This omission is exactly
what `CascadeDecays` calls `CurrentFrame()`, and what `_effectively_at_rest`
tries to detect automatically.

The walk in `WALK_PROGRAMS` *does* keep its root `ToRestFrame((1,2,3,4))` line,
because a pure boost with γ = 1 is exactly the identity — it needs no such
special case.  See §5.
"""
function idt_program(topology, depth)
    topology == "DxD" && depth == 1 &&
        return (MeasureCosThetaPhi(:ang, (1, 2, 3)),)
    topology == "DxD" && depth == 2 &&
        return (ToHelicityFrame((1, 2, 3)), MeasureCosThetaPhi(:ang, (1, 2)))
    topology == "DxD" && depth == 3 &&
        return (ToHelicityFrame((1, 2, 3)), ToHelicityFrame((1, 2)), MeasureCosThetaPhi(:ang, (1,)))
    topology == "dk" && depth == 1 &&
        return (MeasureCosThetaPhi(:ang, (1, 2)),)
    topology == "dk" && depth == 2 &&
        return (ToHelicityFrame((1, 2)), MeasureCosThetaPhi(:ang, (1,)))
    return (ToHelicityFrame((3, 4)), MeasureCosThetaPhi(:ang, (3,)))
end

topology_of(name) = name == "DxD" ? B2DxDK.dxd_topology : B2DxDK.dk_topology

# ---------------------------------------------------------------------------
# 1. Frame trace — the transparency check
# ---------------------------------------------------------------------------

describe(i::PlantLabAxes) = @sprintf("PlantLabAxes(%d)", i.slot)
describe(i::ToRestFrame) = @sprintf("ToRestFrame(%s)", i.indices)
describe(i::TransportAxes) = @sprintf("TransportAxes(%d, %s)", i.slot, i.along)
describe(i::MeasureEulerZXZ) = @sprintf("MeasureEulerZXZ(:%s, %s, %d)", i.tag, i.indices, i.slot)

"""
Run the program one instruction at a time, reporting after each line which
system is at rest and where the carried ẑ points.  If a reader cannot predict
these columns from the program text alone, the program is not transparent.
"""
function trace_program(objs, program)
    state = with_axes(objs)
    resting = nothing
    @printf("    %-34s  %-26s  %s\n", "instruction", "|Σp⃗| of last boosted system", "carried ẑ")
    for instr in program
        (state, _) = apply_decay_instruction(instr, state)
        instr isa ToRestFrame && (resting = instr.indices)
        residual = resting === nothing ? NaN : norm(p3(mom(state, resting)))
        ax = axes_at(state, AX)
        zstr = any(isnan, ax.ẑ) ? "(unplanted)" :
            @sprintf("(% .4f, % .4f, % .4f)", ax.ẑ[1], ax.ẑ[2], ax.ẑ[3])
        @printf("    %-34s  %-26s  %s\n", describe(instr),
            isnan(residual) ? "—" : @sprintf("%.2e", residual), zstr)
    end
    return state
end

# ---------------------------------------------------------------------------

function main()
    event, objs, lab = load_event()
    Bp = sum(objs)
    failures = String[]
    note(ok, msg) = (ok || push!(failures, msg); ok)

    println("=" ^ 78)
    println("CROSS-PRODUCT HELICITY WALK — validation on crosscheck_event.json")
    println("=" ^ 78)
    @printf("External order (D0=1, π=2, D=3, K=4);  |p⃗_B| = %.3e GeV\n", norm(p3(Bp)))
    @printf("CascadeDecays at-rest tolerance: rtol·max(E,1) = %.3e  ⇒  B counts as MOVING,\n",
        1.0e-12 * max(abs(Bp.E), 1.0))
    println("so KinematicPoint silently uses HelicityRootFrame on this event (see §4).\n")

    # --- 1. frame trace ----------------------------------------------------
    println("─" ^ 78)
    println("1. FRAME TRACE — every boost is a program line")
    println("─" ^ 78)
    for (topo, depth, label, _, program) in WALK_PROGRAMS
        depth == 3 || continue          # deepest walk of each topology
        println("\n  $topo, $label")
        trace_program(objs, program)
    end
    println("\n  No rest-frame table is built anywhere: `objs` is the only carrier,")
    println("  and slot $AX holds the (ẑ, x̂) marker that rides along with it.")

    # --- 2. TF-PWA equivalence --------------------------------------------
    println("\n" * "─" ^ 78)
    println("2. TF-PWA EQUIVALENCE — walk vs standalone cal_chain_boost port")
    println("─" ^ 78)
    chains = Dict(
        "DxD" => [(:Bp, [:X_dxd, :K]), (:X_dxd, [:Dst, :D]), (:Dst, [:D0, :pi])],
        "dk" => [(:Bp, [:Dst, :X_dk]), (:Dst, [:D0, :pi]), (:X_dk, [:D, :K])],
    )
    reference = Dict(k => [r for r in last(run_crossproduct_chain(lab, v)) if r.used_in_D]
        for (k, v) in chains)

    @printf("  %-4s %-24s %13s %13s %10s %10s\n", "topo", "vertex", "walk α", "walk β", "Δα", "Δβ")
    walk_results = Dict{Tuple{String, Int}, Any}()
    for (topo, depth, label, _, program) in WALK_PROGRAMS
        (_, res) = apply_decay_instruction(program, with_axes(objs))
        walk_results[(topo, depth)] = res.ang
        ref = reference[topo][depth]
        Δα = abs(wrap_delta(res.ang.α, ref.alpha))
        Δβ = abs(res.ang.β - ref.beta)
        tol = depth == 1 ? TOL_ROOT : TOL_MACHINE
        note(Δα < tol && Δβ < tol, "TF-PWA equivalence $topo depth $depth")
        @printf("  %-4s %-24s %13.9f %13.9f %10.2e %10.2e\n", topo, label, res.ang.α, res.ang.β, Δα, Δβ)
    end
    println("""
  The standalone port is the SAME physics in TF-PWA's own shape (nested rest
  table + per-vertex Euler bundle).  Agreement here means the decomposition
  into program lines lost nothing.
  Root-vertex residual ≈ 3e-10: `FourVectors.Bz` is parameterised by γ, and for
  |β| ≈ 4e-11 the value γ = 1 + 8e-22 rounds to exactly 1.0, so the walk drops a
  boost the reference's β-parameterised formula still applies.  Same at-rest
  pathology as CascadeDecays' `_effectively_at_rest`, seen from the other side.""")

    # --- 3. IDT equivalence -----------------------------------------------
    println("\n" * "─" ^ 78)
    println("3. IDT EQUIVALENCE — carried axes vs realigned frame, same answer")
    println("─" ^ 78)
    println("  Left:  ToRestFrame chain + MeasureEulerZXZ against carried (ẑ, x̂)")
    println("  Right: ToHelicityFrame chain + MeasureCosThetaPhi against (0,0,1)/(1,0,0)")
    println("         (started in the current frame — see idt_program's docstring)\n")
    @printf("  %-4s %-24s %10s %10s\n", "topo", "vertex", "|α−ϕ|", "|β−θ|")
    for (topo, depth, label, _, _) in WALK_PROGRAMS
        (_, res_idt) = apply_decay_instruction(idt_program(topo, depth), objs)
        w = walk_results[(topo, depth)]
        θ = acos(clamp(res_idt.ang.cosθ, -1, 1))
        Δα = abs(wrap_delta(w.α, res_idt.ang.ϕ))
        Δβ = abs(w.β - θ)
        note(Δα < TOL_MACHINE && Δβ < TOL_MACHINE, "IDT equivalence $topo depth $depth")
        @printf("  %-4s %-24s %10.2e %10.2e\n", topo, label, Δα, Δβ)
    end
    println("""
  This is the pedagogical core: the two conventions are the same convention.
  `ToHelicityFrame` spends the alignment rotation and needs no memory;
  `ToRestFrame` keeps the frame still and remembers the rotation as (ẑ, x̂).""")

    # --- 4. CascadeDecays --------------------------------------------------
    println("\n" * "─" ^ 78)
    println("4. CASCADEDECAYS — depth by depth")
    println("─" ^ 78)
    point = KinematicPoint(B2DxDK.kinematic_task, objs)
    @printf("  %-4s %-24s %10s %10s %10s %10s %10s %10s\n", "topo", "vertex",
        "Δϕ curr", "Δθ curr", "Δϕ root", "Δθ root", "Δϕ KPt", "Δθ KPt")
    for (topo, depth, label, address, _) in WALK_PROGRAMS
        w = walk_results[(topo, depth)]
        t = topology_of(topo)
        cols = Float64[]
        for frame in (CurrentFrame(), HelicityRootFrame())
            a = vertex_angles(CascadeKinematics(t, objs; initial_frame = frame), t, address)
            push!(cols, abs(wrap_delta(w.α, a.ϕ)), abs(w.β - acos(clamp(a.cosθ, -1, 1))))
        end
        a = vertex_angles(kinematics_at(point, t), t, address)
        push!(cols, abs(wrap_delta(w.α, a.ϕ)), abs(w.β - acos(clamp(a.cosθ, -1, 1))))
        note(cols[1] < (depth == 1 ? TOL_ROOT : TOL_MACHINE) && cols[2] < TOL_MACHINE,
            "CascadeDecays CurrentFrame $topo depth $depth")
        @printf("  %-4s %-24s %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
            topo, label, cols...)
    end
    println("""
  CurrentFrame: agreement at EVERY depth in BOTH topologies, to ~1e-15.
    The previously reported "depth ≥ 2 divergence, θ ≈ β but ϕ ≠ α" was not
    physics.  It came from `lorentz_boost` in the standalone port returning the
    energy component untransformed (`[v[1]; spatial]`).  The spatial part of
    that one boost stays right, but the next `rest_vector` divides by the stale
    energy to get its β, so everything below depth 2 was boosted with the wrong
    velocity — and the four-vectors stopped conserving mass.  Fixed here and in
    scripts/compare_angles_one_event.jl.
  HelicityRootFrame / KinematicPoint: ϕ off by a constant ≈ 1.92 rad at depth
    ≥ 2 while θ matches — the signature of one spurious azimuthal rotation, not
    a convention mismatch.  CascadeDecays prepends ToHelicityFrame(B) with the
    ϕ, θ of a 2e-10 GeV momentum, i.e. numerical noise.  `ToRestFrame` needs no
    such branch: at γ = 1 it is exactly the identity (see §5).""")

    # --- 5. frame algebra --------------------------------------------------
    println("\n" * "─" ^ 78)
    println("5. FRAME ALGEBRA — the two trailing rotations")
    println("─" ^ 78)
    P = mom(objs, (1, 2, 3))
    θP, ϕP, γP = polar_angle(P), azimuthal_angle(P), boost_gamma(P)
    d_pure = maximum(objs) do p
        norm(collect(boost_to_rest(p, P)) .- collect(p |> Rz(-ϕP) |> Ry(-θP) |> Bz(-γP) |> Ry(θP) |> Rz(ϕP)))
    end
    d_hel = maximum(objs) do p
        norm(collect(transform_to_cmf(p, P)) .- collect(boost_to_rest(p, P) |> Rz(-ϕP) |> Ry(-θP)))
    end
    m_before = [mass(p) for p in objs]
    m_after = [mass(boost_to_rest(p, P)) for p in objs]
    d_mass = maximum(abs.(m_before .- m_after))
    note(d_pure < 1.0e-14, "ToRestFrame is the rotate/boost/rotate-back sandwich")
    note(d_hel < 1.0e-14, "ToHelicityFrame == ToRestFrame then realignment")
    note(d_mass < 1.0e-12, "ToRestFrame conserves mass")
    @printf("  ToRestFrame == Rz(-ϕ)·Ry(-θ)·Bz(-γ)·Ry(θ)·Rz(ϕ)      max dev %.2e\n", d_pure)
    @printf("  ToHelicityFrame == ToRestFrame |> Rz(-ϕ) |> Ry(-θ)  max dev %.2e\n", d_hel)
    @printf("  masses conserved by ToRestFrame                     max dev %.2e\n", d_mass)
    println("""
  The second line is the whole story: `ToHelicityFrame` is `ToRestFrame` plus a
  realignment.  The cross-product method is what you get by NOT applying that
  realignment and tracking it as (ẑ, x̂) instead.""")

    # --- summary -----------------------------------------------------------
    println("\n" * "=" ^ 78)
    if isempty(failures)
        println("ALL CHECKS PASSED")
    else
        println("FAILURES (", length(failures), "):")
        foreach(f -> println("  • ", f), failures)
    end
    println("=" ^ 78)
    return isempty(failures)
end

main() || exit(1)
