"""
    CrossProductWalk

TF-PWA's cross-product helicity angles, written as an **InstructionalDecayTrees
program** — a decay walk you can read top to bottom.

Physics content is identical to `TFPWACrossProductHelicity.jl` (the direct port
of TF-PWA `cal_chain_boost` + `EulerAngle.angle_zx_z_getx`), but the program
structure is different, and that is the whole point.  See
`DESIGN_crossproduct_idt.md` for why the direct port is *not* an IDT tutorial.

# The one idea

`InstructionalDecayTrees.ToHelicityFrame(P)` is

    p |> Rz(-ϕ) |> Ry(-θ) |> Bz(-γ)          # boost, then realign axes onto P

TF-PWA instead uses a **pure boost**, which is the same thing with the
alignment rotation undone at the end:

    p |> Rz(-ϕ) |> Ry(-θ) |> Bz(-γ) |> Ry(θ) |> Rz(ϕ)

Those two trailing rotations are the *entire* difference between the two
frameworks.  `ToHelicityFrame` spends them, so its ẑ is always `(0,0,1)` and
nothing has to be remembered.  [`ToRestFrame`](@ref) keeps them, so the frame
never rotates — and the price is that you must carry the helicity axes `(ẑ, x̂)`
along as data.  That is what [`TransportAxes`](@ref) does, and
[`MeasureEulerZXZ`](@ref) is `MeasureCosThetaPhi` against those carried axes
instead of against `(0,0,1)`/`(1,0,0)`.

# State

`objs` is the *single* frame carrier, exactly as in IDT.  Slots `1:N` hold the
final-state four-vectors; one extra slot holds a [`HelicityAxes`](@ref) marker.
Build it with [`with_axes`](@ref).  After every instruction, **everything in
`objs` — four-vectors and axes alike — refers to the same frame.**

# Instruction set

| instruction                       | kind      | does                                              |
|-----------------------------------|-----------|---------------------------------------------------|
| [`PlantLabAxes`](@ref)            | state     | writes the ẑ=(0,0,1), x̂=(1,0,0) convention        |
| [`ToRestFrame`](@ref)             | transform | pure boost into the rest frame of `indices`        |
| [`TransportAxes`](@ref)           | transform | carries `(ẑ, x̂)` across one decay vertex           |
| [`MeasureEulerZXZ`](@ref)         | measure   | `(α, β)` of `indices` against the carried axes     |

Programs run through the stock IDT driver, `apply_decay_instruction(program, objs)`.

# Example — DxD topology, vertex `D*→ D⁰ π`, external order `(D0=1, π=2, D=3, K=4)`

```julia
program = (
    PlantLabAxes(5),                  # ẑ=(0,0,1), x̂=(1,0,0)         [axes planted]
    ToRestFrame((1, 2, 3, 4)),        # pure boost                    [B rest frame]
    TransportAxes(5, (1, 2, 3)),      # B → X K vertex                [axes ride onto X]
    ToRestFrame((1, 2, 3)),           # pure boost                    [X rest frame]
    TransportAxes(5, (1, 2)),         # X → D* D vertex               [axes ride onto D*]
    ToRestFrame((1, 2)),              # pure boost                    [D* rest frame]
    MeasureEulerZXZ(:v3, (1,), 5),    # D* → D⁰ π: the (α,β) in the Wigner D
)
(_, res) = apply_decay_instruction(program, with_axes(objs))
res.v3   # (α, β, γ, cosβ)
```
"""
module CrossProductWalk

using FourVectors
using InstructionalDecayTrees
using LinearAlgebra
using StaticArrays

import InstructionalDecayTrees: AbstractInstruction, AbstractMeasureInstruction,
    apply_decay_instruction

export HelicityAxes, with_axes, axes_at
export PlantLabAxes, ToRestFrame, TransportAxes, MeasureEulerZXZ
export boost_to_rest, euler_zxz, transported_x, cross_unit, momentum_of

const Vec3 = SVector{3, Float64}

"""Degeneracy threshold of TF-PWA `EulerAngle.cross_unit`."""
const AXIS_EPS = 1.0e-12

# ---------------------------------------------------------------------------
# 1. The extra state: a helicity-axis marker that lives inside `objs`
# ---------------------------------------------------------------------------

"""
    HelicityAxes(ẑ, x̂)

A pair of orthonormal direction markers occupying one slot of `objs`.

It is *not* a four-vector and must never be summed with one; it records where
the helicity ẑ and x̂ of the current decay step point, **in the coordinates of
whatever frame `objs` is currently in**.

`HelicityAxes()` is the "not yet planted" marker (`NaN`), so forgetting
[`PlantLabAxes`](@ref) fails loudly instead of silently using stale axes.
"""
struct HelicityAxes
    ẑ::Vec3
    x̂::Vec3
end

HelicityAxes() = HelicityAxes(Vec3(NaN, NaN, NaN), Vec3(NaN, NaN, NaN))

Base.show(io::IO, a::HelicityAxes) =
    print(io, "HelicityAxes(ẑ=", round.(a.ẑ; digits = 6), ", x̂=", round.(a.x̂; digits = 6), ")")

"""
    with_axes(objs, naxes = 1)

Append `naxes` unplanted [`HelicityAxes`](@ref) slots to `objs`.  Particle
indices `1:length(objs)` are untouched, so a program written against the bare
four-vectors keeps working.  This is state construction, not physics — the
convention itself is chosen by [`PlantLabAxes`](@ref) inside the program.
"""
with_axes(objs, naxes::Int = 1) = (objs..., ntuple(_ -> HelicityAxes(), naxes)...)

"""Read the axis marker in slot `slot` (convenience for inspecting a run)."""
axes_at(objs, slot::Int) = objs[slot]::HelicityAxes

_setslot(objs::Tuple, slot::Int, value) =
    ntuple(k -> k == slot ? value : objs[k], length(objs))

# ---------------------------------------------------------------------------
# 2. Geometry helpers (plain functions — deliberately NOT instructions)
# ---------------------------------------------------------------------------

"""
    cross_unit(a, b)

Unit normal of `a` and `b`, with TF-PWA's tie-break for the collinear case
(`EulerAngle.cross_unit`): when `a ∥ b` the normal is undefined, and TF-PWA
picks a reproducible one by nudging `b` along `(1,1,1)`.
"""
function cross_unit(a, b)
    c = cross(a, b)
    if norm(c) < AXIS_EPS
        c = cross(a, Vec3(1, 1, 1) .+ b)
    end
    return Vec3(normalize(c))
end

"""Spatial direction of a four-vector, as a unit 3-vector."""
direction(p::FourVector) = Vec3(normalize(SVector(p.px, p.py, p.pz)))

"""
    euler_zxz(ẑ₁, x̂₁, ẑ₂) -> (α, β, γ)

Euler angles of the rotation `Rz(α) Ry(β) Rz(γ)` that carries the tracked triad
`(x̂₁, ŷ₁, ẑ₁)` onto a triad whose z-axis is `ẑ₂` (TF-PWA
`EulerAngle.angle_zx_z_getx`).  `γ ≡ 0` by construction: the rotation is taken
in the `(ẑ₁, ẑ₂)` plane, which is the helicity convention.

`α` and `β` are just an azimuth and a polar angle — the only difference from
`MeasureCosThetaPhi` is that they are measured against the *carried* axes
`(ẑ₁, x̂₁)` rather than against the coordinate axes `(0,0,1)`/`(1,0,0)`.  When
the carried axes happen to be the coordinate axes, `α = ϕ` and `β = θ`.
"""
function euler_zxz(ẑ₁, x̂₁, ẑ₂)
    u_ẑ₁ = Vec3(normalize(ẑ₁))
    u_ẑ₂ = Vec3(normalize(ẑ₂))
    ŷ₁ = cross_unit(u_ẑ₁, x̂₁)             # complete the carried triad
    u_x̂₁ = cross_unit(ŷ₁, u_ẑ₁)           # re-orthogonalise x̂₁ against ẑ₁
    ŷ_r = cross_unit(u_ẑ₁, u_ẑ₂)          # normal of the (ẑ₁, ẑ₂) plane
    x̂_r = cross_unit(ŷ_r, u_ẑ₁)           # in-plane companion of ẑ₁
    α = atan(x̂_r ⋅ ŷ₁, x̂_r ⋅ u_x̂₁)        # azimuth of the decay plane about ẑ₁
    β = atan(u_ẑ₂ ⋅ x̂_r, u_ẑ₂ ⋅ u_ẑ₁)     # polar angle of ẑ₂ away from ẑ₁
    return (α = α, β = β, γ = 0.0)
end

"""
    transported_x(ẑ₁, ẑ₂) -> x̂₂

The new x̂ after the helicity frame is carried from `ẑ₁` onto `ẑ₂`: rotate about
the `(ẑ₁, ẑ₂)` plane normal, i.e. `x̂₂ = ŷ_r × ẑ₂`.  Independent of `x̂₁` — the
azimuth is fixed by the decay plane, not by where x̂ used to point.
"""
transported_x(ẑ₁, ẑ₂) = cross_unit(cross_unit(ẑ₁, ẑ₂), ẑ₂)

"""
    momentum_of(objs, indices)

Sum of the four-vectors at `indices` (negative index ⇒ minus that four-vector),
same rule as `InstructionalDecayTrees.get_fourvector`.  Errors if an axis slot
is addressed — axis markers are not four-vectors and cannot be summed.
"""
function momentum_of(objs, indices::Tuple{Vararg{Int}})
    for i in indices
        objs[abs(i)] isa FourVector ||
            error("momentum_of: slot $(abs(i)) holds $(typeof(objs[abs(i)])), not a FourVector")
    end
    length(indices) == 1 && return (i = only(indices); i < 0 ? -objs[-i] : objs[i])
    return sum(i < 0 ? -objs[-i] : objs[i] for i in indices)
end

# ---------------------------------------------------------------------------
# 3. The pure boost, and how axis markers respond to it
# ---------------------------------------------------------------------------

"""
    boost_to_rest(p, P)

Pure boost of the four-vector `p` into the rest frame of `P`: rotate `P` onto
+ẑ, boost along ẑ, rotate back.  The trailing `Ry(θ) |> Rz(ϕ)` is what makes it
*pure* — drop those two and this is `FourVectors.transform_to_cmf`, i.e.
`ToHelicityFrame`.

Aside: because the round trip is exact, this is also numerically safe when `P`
is already at rest (`γ == 1` ⇒ `Bz` is the identity ⇒ the rotations cancel),
whatever garbage `polar_angle`/`azimuthal_angle` return for a null 3-momentum.
`ToHelicityFrame` has no such protection, which is why `CascadeDecays` needs
`_effectively_at_rest` / `CurrentFrame` at the root.
"""
function boost_to_rest(p::FourVector, P::FourVector)
    θ = polar_angle(P)
    ϕ = azimuthal_angle(P)
    γ = boost_gamma(P)
    return p |> Rz(-ϕ) |> Ry(-θ) |> Bz(-γ) |> Ry(θ) |> Rz(ϕ)
end

"""
A direction marker is **inert** under a pure boost.

A pure boost relates two frames whose spatial axes are parallel by definition,
so a marker that records "the previous helicity ẑ pointed *there*" keeps the
same three numbers.  This one line is why the axes may live in `objs` next to
the four-vectors without ever leaving the common frame — and it is exactly the
assumption TF-PWA makes when it reuses `set_z[core]` after `cal_chain_boost`.

(Under a *rotation* the marker would rotate like any 3-vector.  No instruction
here rotates the frame, so that case never arises.)
"""
boost_to_rest(a::HelicityAxes, ::FourVector) = a

# ---------------------------------------------------------------------------
# 4. Instructions
# ---------------------------------------------------------------------------

"""
    PlantLabAxes(slot)

Write the starting helicity convention into `slot`: `ẑ = (0,0,1)`, `x̂ = (1,0,0)`.

This is where the lab-axis convention enters the program, and it is the
cross-product counterpart of "which frame do we start in?".  It is a state
write, not a transform: `objs` are not moved.
"""
struct PlantLabAxes <: AbstractInstruction
    slot::Int
end

function apply_decay_instruction(instr::PlantLabAxes, objs)
    planted = HelicityAxes(Vec3(0, 0, 1), Vec3(1, 0, 0))
    return (_setslot(objs, instr.slot, planted), (;))
end

"""
    ToRestFrame(indices)

Pure boost of everything in `objs` into the rest frame of `sum(objs[indices])`.

The IDT counterpart is `ToHelicityFrame`, which additionally rotates the frame
so the boost direction becomes +ẑ.  `ToRestFrame` deliberately does not: it
leaves the spatial axes where they were, so `(ẑ, x̂)` markers carried in `objs`
stay valid without being touched (see [`boost_to_rest`](@ref)).

Four-vectors move; axis markers do not.  Both end up in the new frame.
"""
struct ToRestFrame{T <: Tuple} <: AbstractInstruction
    indices::T
    ToRestFrame(indices::Tuple{Vararg{Int}}) = new{typeof(indices)}(indices)
end
ToRestFrame(indices::Int...) = ToRestFrame(indices)
ToRestFrame(indices::AbstractVector{<:Integer}) = ToRestFrame(Tuple(Int.(indices)))

function apply_decay_instruction(instr::ToRestFrame, objs)
    P = momentum_of(objs, instr.indices)
    return (map(o -> boost_to_rest(o, P), objs), (;))
end

"""
    TransportAxes(slot, along)

Carry the helicity axes in `slot` across one decay vertex: the new ẑ is the
direction of `sum(objs[along])` in the current frame, and the new x̂ is fixed by
the decay plane (see [`transported_x`](@ref)).

Run it **while still in the parent rest frame** — `along` must be the daughter's
momentum measured there, exactly as in TF-PWA's `set_z[out] = data[core][out]`.
The subsequent `ToRestFrame(along)` then leaves the fresh axes untouched.

This is the instruction to reach for when a vertex is only on the *path* to the
angle you want.  TF-PWA computes `(α, β)` at such vertices too, but
`HelicityDecay.get_D_matrix_term` reads only `data[outs[0]]["ang"]`, so those
numbers never enter an amplitude.  Splitting transport from measurement makes
that visible: a vertex you pass through gets `TransportAxes`, a vertex you use
gets [`MeasureEulerZXZ`](@ref).
"""
struct TransportAxes{T <: Tuple} <: AbstractInstruction
    slot::Int
    along::T
    TransportAxes(slot::Int, along::Tuple{Vararg{Int}}) = new{typeof(along)}(slot, along)
end
TransportAxes(slot::Int, along::Int...) = TransportAxes(slot, along)

function apply_decay_instruction(instr::TransportAxes, objs)
    old = axes_at(objs, instr.slot)
    ẑ₂ = direction(momentum_of(objs, instr.along))
    carried = HelicityAxes(ẑ₂, transported_x(old.ẑ, ẑ₂))
    return (_setslot(objs, instr.slot, carried), (;))
end

"""
    MeasureEulerZXZ(tag, indices, slot; branch = 1)

Measure the Euler angles `(α, β, γ=0)` of `sum(objs[indices])` against the axes
carried in `slot`, and store them under `tag` as `(α, β, γ, cosβ)`.

This is `MeasureCosThetaPhi` with the reference axes supplied explicitly:
`α` plays the role of `ϕ` and `β` of `θ`.  If the carried axes are the
coordinate axes, the two agree exactly.

`branch` reproduces TF-PWA's per-daughter `bias`: daughter `n` folds `α` into
`[-nπ, (2-n)π)`.  Only `branch = 1` (the first-listed daughter) ever reaches a
Wigner D, so the default is 1 and the keyword exists mainly to document that the
other branch is a bookkeeping convention, not different physics.
"""
struct MeasureEulerZXZ{T <: Tuple} <: AbstractMeasureInstruction
    tag::Symbol
    indices::T
    slot::Int
    branch::Int
    function MeasureEulerZXZ(tag::Symbol, indices::Tuple{Vararg{Int}}, slot::Int; branch::Int = 1)
        return new{typeof(indices)}(tag, indices, slot, branch)
    end
end
MeasureEulerZXZ(tag::Symbol, index::Int, slot::Int; branch::Int = 1) =
    MeasureEulerZXZ(tag, (index,), slot; branch)

"""TF-PWA folds daughter `n`'s azimuth into `[-nπ, (2-n)π)` (`bias = -nπ`)."""
_branch_fold(α, branch::Int) = (bias = -branch * π; mod(α - bias, 2π) + bias)

function apply_decay_instruction(instr::MeasureEulerZXZ, objs)
    ax = axes_at(objs, instr.slot)
    ẑ₂ = direction(momentum_of(objs, instr.indices))
    ang = euler_zxz(ax.ẑ, ax.x̂, ẑ₂)
    value = (
        α = _branch_fold(ang.α, instr.branch),
        β = ang.β,
        γ = ang.γ,
        cosβ = cos(ang.β),
    )
    return (objs, NamedTuple{(instr.tag,)}((value,)))
end

end # module CrossProductWalk
