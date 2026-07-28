"""
TF-PWA cross-product helicity angles, in TF-PWA's own shape.

**Two roles, both deliberate.**

1. *Reference implementation.* A direct port of `cal_chain_boost` +
   `EulerAngle.angle_zx_z_getx`: nested rest-frame boosts into a `Dict`, then a
   per-vertex Euler bundle. `run_crossproduct_walk.jl` §2 checks the IDT program
   in `CrossProductWalk.jl` against it.

2. *Negative example.* This file is what an IDT tutorial must **not** look like.
   `BoostRestTable()` hides an entire boost phase behind one instruction;
   `CrossProductVertex` hides a loop, the bias convention and the axis update
   behind another; the physics lives in the `rest` dict and `lab_p4` while
   `objs` never move. Numerically correct, pedagogically empty. See
   `DESIGN_crossproduct_idt.md` §6.

Fixed 2026-07-28: `lorentz_boost` returned the energy component untransformed,
which corrupted every boost below depth 2 (the next `rest_vector` divides by
that energy to get its β) and broke mass conservation. It had been reported as
a CascadeDecays↔TF-PWA convention mismatch. See `DESIGN_crossproduct_idt.md` §5.
"""
module TFPWACrossProductHelicity

using FourVectors
using LinearAlgebra
using StaticArrays
using Printf

export angle_zx_z_getx,
    HelicityAxisState,
    build_rest_table!,
    run_crossproduct_chain,
    apply_instruction!,
    InitLabAxes,
    BoostRestTable,
    CrossProductVertex,
    LocalToRestFrame,
    MeasureSphericalInFrame,
    IDTExperimentState,
    init_experiment,
    get_fourvector_from_objs

const SVec3 = SVector{3, Float64}
const EPS = 1.0e-12

# ---------------------------------------------------------------------------
# TF-PWA boost + cross-product core
# ---------------------------------------------------------------------------

unitvec(v) = SVec3(v ./ norm(v))

function cross_unit(a, b)
    c = cross(a, b)
    n = norm(c)
    n < EPS && (c = cross(a, SVec3(1, 1, 1) .+ b); n = norm(c))
    return unitvec(c)
end

angle_from(v, x, y) = atan(v ⋅ y, v ⋅ x)

"""
    angle_zx_z_getx(z1, x1, z2) -> ((alpha,beta,gamma), x2)

Cross-product Euler construction (TF-PWA `EulerAngle.angle_zx_z_getx`).
"""
function angle_zx_z_getx(z1, x1, z2)
    u_z1 = unitvec(z1)
    u_z2 = unitvec(z2)
    u_y1 = cross_unit(u_z1, x1)
    u_x1 = cross_unit(u_y1, u_z1)
    u_yr = cross_unit(u_z1, u_z2)
    u_xr = cross_unit(u_yr, u_z1)
    alpha = angle_from(u_xr, u_x1, u_y1)
    beta = angle_from(u_z2, u_z1, u_xr)
    u_x2 = cross_unit(u_yr, u_z2)
    return (alpha=alpha, beta=beta, gamma=0.0), u_x2
end

boost_vector(v) = SVec3(v[2], v[3], v[4]) / v[1]

function lorentz_boost(v::Vector{Float64}, beta::SVec3)
    beta2 = dot(beta, beta)
    gamma = 1.0 / sqrt(1.0 - beta2)
    bp = dot(beta, SVec3(v[2], v[3], v[4]))
    gamma2 = beta2 > EPS ? (gamma - 1.0) / beta2 : 0.0
    spatial = SVec3(v[2], v[3], v[4]) .+ gamma2 .* bp .* beta .+ gamma .* v[1] .* beta
    energy = gamma * (v[1] + bp)
    return [energy; collect(spatial)]
end

rest_vector(core, other) = lorentz_boost(other, -boost_vector(core))

"""
    HelicityAxisState

Sidecar for TF-PWA axis transport + nested rest-frame momenta.
"""
mutable struct HelicityAxisState
    rest::Dict{Symbol, Dict{Symbol, Vector{Float64}}}
    axes_z::Dict{Symbol, SVec3}
    axes_x::Dict{Symbol, SVec3}
end

HelicityAxisState() = HelicityAxisState(Dict(), Dict(), Dict())

function build_rest_table!(state::HelicityAxisState, lab_p4::Dict{Symbol, Vector{Float64}}, chain)
    particle_set = Set{Symbol}()
    for (_, outs) in chain
        foreach(x -> push!(particle_set, x), outs)
    end
    core_parent = Dict{Symbol, Symbol}()   # daughter → immediate parent core
    state.rest = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()
    pending = Vector{eltype(chain)}(chain)
    while !isempty(pending)
        next_pending = eltype(chain)[]
        for (core, outs) in pending
            if core == :Bp
                p_rest = lab_p4[:Bp]
                bucket = Dict{Symbol, Vector{Float64}}()
                for out in outs
                    core_parent[out] = core
                    bucket[out] = rest_vector(p_rest, lab_p4[out])
                end
                for name in particle_set
                    haskey(bucket, name) || (bucket[name] = rest_vector(p_rest, lab_p4[name]))
                end
                state.rest[core] = bucket
            elseif haskey(core_parent, core)
                parent = core_parent[core]
                p_rest = state.rest[parent][core]
                bucket = Dict{Symbol, Vector{Float64}}()
                for out in outs
                    core_parent[out] = core
                    bucket[out] = rest_vector(p_rest, state.rest[parent][out])
                end
                for name in particle_set
                    haskey(bucket, name) || (bucket[name] = rest_vector(p_rest, state.rest[parent][name]))
                end
                state.rest[core] = bucket
            else
                push!(next_pending, (core, outs))
            end
        end
        isempty(next_pending) && break
        pending = next_pending
        length(pending) >= length(chain) + 1 && error("build_rest_table!: unresolved cores $pending")
    end
    return state
end

function init_lab_axes!(state::HelicityAxisState, root::Symbol = :Bp)
    state.axes_z[root] = SVec3(0, 0, 1)
    state.axes_x[root] = SVec3(1, 0, 0)
    return state
end

function crossproduct_vertex_angles!(state::HelicityAxisState, core::Symbol, outs::Vector{Symbol})
    results = NamedTuple[]
    bias = -pi
    for (i, out) in enumerate(outs)
        z2 = SVec3(state.rest[core][out][2], state.rest[core][out][3], state.rest[core][out][4])
        ang, x2 = angle_zx_z_getx(state.axes_z[core], state.axes_x[core], z2)
        alpha = mod(ang.alpha - bias, 2pi) + bias
        bias -= pi
        state.axes_z[out] = unitvec(z2)
        state.axes_x[out] = x2
        push!(results, (
            core = core,
            out = out,
            alpha = alpha,
            beta = ang.beta,
            gamma = ang.gamma,
            used_in_D = i == 1,
        ))
    end
    return results
end

function run_crossproduct_chain(lab_p4::Dict{Symbol, Vector{Float64}}, chain)
    state = HelicityAxisState()
    build_rest_table!(state, lab_p4, chain)
    init_lab_axes!(state, :Bp)
    all_results = typeof((core=:Bp, out=:K, alpha=0.0, beta=0.0, gamma=0.0, used_in_D=true))[]
    for (core, outs) in chain
        append!(all_results, crossproduct_vertex_angles!(state, core, collect(Symbol, outs)))
    end
    return state, all_results
end

# ---------------------------------------------------------------------------
# Local IDT-style bridge (Option B): sidecar + objs, no package extension
# ---------------------------------------------------------------------------

struct InitLabAxes end
struct BoostRestTable end
struct CrossProductVertex{C,O}
    core::C
    outs::O
end
CrossProductVertex(core, outs...) = CrossProductVertex(core, outs)

"""Boost all objs to rest frame of sum(indices), rotation-free (Bz only)."""
struct LocalToRestFrame{T<:Tuple} end
LocalToRestFrame(indices...) = LocalToRestFrame(indices)

struct MeasureSphericalInFrame{I}
    indices::I
end
MeasureSphericalInFrame(i) = MeasureSphericalInFrame((i,))

struct IDTExperimentState
    objs
    axis::HelicityAxisState
    lab_p4::Dict{Symbol, Vector{Float64}}
    chain
    measurements::Vector{NamedTuple}
end

function get_fourvector_from_objs(objs, indices::Tuple{Vararg{Int}})
    if length(indices) == 1
        idx = only(indices)
        return idx < 0 ? -objs[-idx] : objs[idx]
    end
    return sum(idx < 0 ? -objs[-idx] : objs[idx] for idx in indices)
end

function init_experiment(objs, lab_p4, chain)
    return IDTExperimentState(objs, HelicityAxisState(), lab_p4, chain, NamedTuple[])
end

function apply_instruction!(exp::IDTExperimentState, ::InitLabAxes)
    init_lab_axes!(exp.axis, :Bp)
    return exp
end

function apply_instruction!(exp::IDTExperimentState, ::BoostRestTable)
    build_rest_table!(exp.axis, exp.lab_p4, exp.chain)
    return exp
end

function apply_instruction!(exp::IDTExperimentState, instr::CrossProductVertex)
    outs = collect(Symbol, instr.outs)
    results = crossproduct_vertex_angles!(exp.axis, instr.core, outs)
    append!(exp.measurements, results)
    return exp
end

function apply_instruction!(exp::IDTExperimentState, instr::LocalToRestFrame)
    P = get_fourvector_from_objs(exp.objs, instr.indices)
    γ = boost_gamma(P)
    exp.objs = map(p -> Bz(p, -γ), exp.objs)
    return exp
end

function apply_instruction!(exp::IDTExperimentState, instr::MeasureSphericalInFrame)
    p = get_fourvector_from_objs(exp.objs, instr.indices)
    θ = polar_angle(p)
    ϕ = azimuthal_angle(p)
    push!(exp.measurements, (kind = :spherical, θ = θ, ϕ = ϕ, indices = instr.indices))
    return exp
end

apply_instruction!(exp, instr) = apply_instruction!(exp, instr)

"""Run a tuple of local instructions (IDT program style)."""
function run_program!(exp::IDTExperimentState, program)
    for instr in program
        apply_instruction!(exp, instr)
    end
    return exp
end

end # module
