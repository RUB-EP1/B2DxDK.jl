# Cross-product helicity angles as an IDT program — design note

**Status:** the instruction set is being upstreamed to
`InstructionalDecayTrees.jl`. `scripts/CrossProductWalk.jl` is a temporary local
copy so these scripts run before that lands; delete it once IDT ships the
instructions. Nothing has been added to `CascadeDecays.jl`.

**Files:** `CrossProductWalk.jl` (instruction set) · `run_crossproduct_walk.jl`
(single-event validation) · `run_crossproduct_scan.jl` (N-point scan) ·
`TFPWACrossProductHelicity.jl` (TF-PWA-shaped reference, kept deliberately as the
negative example of §6).

---

## 1. The one idea

`InstructionalDecayTrees.ToHelicityFrame` is, in full:

```julia
transform_to_cmf(p, P) = p |> Rz(-ϕ) |> Ry(-θ) |> Bz(-γ)      # ϕ, θ, γ of P
```

TF-PWA never uses this. It boosts with a **pure boost**, which is the same
transformation with the alignment rotation undone at the end:

```julia
boost_to_rest(p, P)     = p |> Rz(-ϕ) |> Ry(-θ) |> Bz(-γ) |> Ry(θ) |> Rz(ϕ)
```

Those two trailing rotations are the entire difference between the two
frameworks, and everything else follows from them:

| | `ToHelicityFrame` (CascadeDecays) | `ToRestFrame` (TF-PWA) |
|---|---|---|
| frame axes | realigned onto the parent each step | never rotate |
| helicity ẑ | always `(0,0,1)` — nothing to remember | drifts — must be carried |
| azimuth measured against | coordinate x̂ = `(1,0,0)` | carried x̂ |
| at the root, parent at rest | rotates by the ϕ, θ of numerical noise | exactly the identity |

So the cross-product method is not a different convention. It is the *same*
convention with the alignment rotation **stored as data instead of applied**.
`run_crossproduct_walk.jl` §3 confirms this: measuring against carried axes and
measuring against realigned axes agree to ~1e-15 at every vertex of both
topologies.

That framing is what makes an IDT expression possible at all. The previous
prototype treated the two as unrelated algorithms and ended up transliterating
one into the other's syntax (§6).

## 2. State: `objs` and nothing else

```julia
objs = (p_D0, p_π, p_D, p_K, HelicityAxes(ẑ, x̂))
#       └────── slots 1:4, four-vectors ─────┘  └── slot 5 ──┘
```

The brief anticipated an unavoidable `(ẑ, x̂)` sidecar, and the retrospective on
the first prototype put it as *"cross-product axes cannot live in boosted
4-vectors correctly."* That is true only if you store them **as** four-vectors:
a spacelike `(0; n̂)` would be boosted, and a boosted direction marker is wrong.

The fix is to give the axes their own type with its own transformation rule:

```julia
boost_to_rest(a::HelicityAxes, ::FourVector) = a
```

A pure boost relates two frames whose spatial axes are parallel *by definition*,
so a marker recording "the previous helicity ẑ pointed there" keeps the same
three numbers. This is precisely the assumption TF-PWA makes when it reuses
`set_z[core]` after `cal_chain_boost` — but there it is implicit in the data
layout, whereas here it is one line you can read and disbelieve.

Consequences:

- **Sidecar count is zero.** `objs` is the single frame carrier, literally.
- The IDT invariant holds verbatim: after every instruction, everything in
  `objs` — four-vectors and axes alike — is expressed in the same frame.
- Particle indices `1:N` are untouched, so `momentum_of` behaves exactly like
  `InstructionalDecayTrees.get_fourvector`. It raises if an axis slot is
  addressed; axis markers are not four-vectors and must never be summed.

Under a *rotation* the marker would rotate like any 3-vector. No instruction in
this set rotates the frame, so that case never arises — but it is the reason
the axes need a type rather than being ordinary data.

## 3. Instruction set

Four instructions. Each is one inspectable act.

| instruction | kind | does |
|---|---|---|
| `PlantLabAxes(slot)` | state write | installs the ẑ=(0,0,1), x̂=(1,0,0) convention |
| `ToRestFrame(indices)` | transform | pure boost into the rest frame of `indices` |
| `TransportAxes(slot, along)` | transform | carries `(ẑ, x̂)` across one decay vertex |
| `MeasureEulerZXZ(tag, indices, slot)` | measure | `(α, β)` of `indices` against carried axes |

They subclass `IDT.AbstractInstruction` / `AbstractMeasureInstruction` and add
methods to `IDT.apply_decay_instruction`, so they run under the stock IDT driver
and compose with `CompositeInstruction` — no parallel executor.

A full program, `DxD` topology, vertex `D* → D⁰ π`:

```julia
(
    PlantLabAxes(5),                  # ẑ=(0,0,1), x̂=(1,0,0)      [axes planted]
    ToRestFrame((1, 2, 3, 4)),        # pure boost                 [B rest frame]
    TransportAxes(5, (1, 2, 3)),      # B → X K vertex             [axes ride onto X]
    ToRestFrame((1, 2, 3)),           # pure boost                 [X rest frame]
    TransportAxes(5, (1, 2)),         # X → D* D vertex            [axes ride onto D*]
    ToRestFrame((1, 2)),              # pure boost                 [D* rest frame]
    MeasureEulerZXZ(:v3, (1,), 5),    # D* → D⁰ π: the (α,β) in the Wigner D
)
```

Program order is the decay walk. Every boost is a line. §1 of the validation
script traces the run instruction-by-instruction, printing the residual momentum
of the system just boosted into (→ 0) and the carried ẑ, so the claim
"a reader can predict the frame after each line" is checkable rather than
asserted.

### Design decisions

**One program per vertex, restarting from the root.** This is `CascadeDecays`'
own idiom (`helicity_angle_program` walks root→vertex for each vertex
separately). It costs a few redundant boosts and buys a straight-line program
with no backtracking instruction, which matters because the decay is a *tree*:
in `dk`, `D*` and `X` are siblings, and a single linear pass through `objs`
cannot visit both without an "un-boost" step that has no physical reading.

**One axis slot, overwritten.** A straight-line walk only ever needs the axes of
the core it is sitting in. The retrospective's schematic had `TransportAxes(1)`
*and* `TransportAxes(2)` at each vertex; with per-vertex programs only the
daughter actually on the path is transported. In `dk` depth 3 that daughter is
the *second* one, `X = (D, K)` — same instruction, no special case, because the
per-daughter `bias` only ever folds a *measured* azimuth and never touches a
transported axis.

**`angle_zx_z_getx` is split in two, not into six.** The physics seam is
measurement vs state update — `(α, β)` enter a Wigner D, `x̂₂` continues the
walk — and those are different *kinds* of thing in IDT. Splitting further, into
separate cross / normalize / atan2 instructions, would be splitting arithmetic,
not physics; `cross_unit`, `euler_zxz` and `transported_x` are documented plain
functions. The split also makes the "used in D" rule structural rather than a
comment: a vertex you pass through gets `TransportAxes`, a vertex you use gets
`MeasureEulerZXZ`. TF-PWA computes `(α, β)` at every vertex, but
`HelicityDecay.get_D_matrix_term` reads only `data[outs[0]]["ang"]`, so the rest
never reach an amplitude.

**`MeasureEulerZXZ` is `MeasureCosThetaPhi` with explicit reference axes.**
`α` plays the role of `ϕ`, `β` of `θ`. When the carried axes happen to be the
coordinate axes the two agree exactly — which is why the root vertex needs no
special handling.

## 4. The initial frame

The walk keeps its root `ToRestFrame((1,2,3,4))` line unconditionally. It is
safe: at `γ == 1` the `Bz` is the identity and the sandwich rotations cancel
exactly, whatever garbage `polar_angle`/`azimuthal_angle` return for a null
3-momentum. Verified to 0.00e+00 in §5.

`ToHelicityFrame` has no such protection — it *applies* those noise angles. That
is what `CascadeDecays._effectively_at_rest` (rtol 1e-12) exists to dodge, and
on `crosscheck_event.json` it does not fire: `|p⃗_B| = 2.1e-10` against a
threshold of `5.3e-12`, so `KinematicPoint` prepends `ToHelicityFrame(B)` and
rotates the whole event. §4 of the validation shows the result — a constant
≈ 1.92 rad ϕ offset at depth ≥ 2 with θ untouched, the signature of one spurious
azimuthal rotation.

The recommendation for the tutorial is therefore *not* to teach a `CurrentFrame`
/ `HelicityRootFrame` switch, but to point out that the switch is a workaround
for a rotation the pure boost never performs. Fixing the tolerance upstream
remains out of scope; this is a note, not a patch.

## 5. What the validation established

Beyond the checks in the brief, one substantive physics correction.

**The reported "depth ≥ 2 divergence, θ ≈ β but ϕ ≠ α" was a bug, not a
convention mismatch.** `lorentz_boost` in `TFPWACrossProductHelicity.jl` (and in
`scripts/compare_angles_one_event.jl`) returned the energy component
untransformed:

```julia
return [v[1]; collect(spatial)]        # should be γ(E + β⃗·p⃗)
```

The spatial part of any single boost is still correct, so depth 1 is exact and
the error is invisible there. One level down, `rest_vector` divides by that
stale energy to get its β — at `X → D*` it used 2.1086 instead of 2.1500 GeV, a
2% velocity error — and the resulting four-vectors stopped conserving mass
(D⁰ came out at 1.9678 instead of 1.8648 GeV). It surfaced as Δβ = 3.8 mrad at
`D* → D⁰`, which the brief had recorded as the *best* CD↔TF agreement.

Both files are fixed. With the fix, `CascadeDecays` and the cross-product method
agree **to ~1e-15 at every vertex of both topologies** under `CurrentFrame`, and
the DPD reference values in `crosscheck_event.json` match both exactly.

The one residual is at the root: ≈ 3e-10 between the walk and the TF-PWA-shaped
reference. `FourVectors.Bz` is parameterised by γ, and for `|β| ≈ 4e-11` the
value `γ = 1 + 8e-22` rounds to exactly 1.0, so the walk drops a boost that the
reference's β-parameterised formula still applies. Well inside the brief's
1e-8 criterion, and the same at-rest pathology as §4 seen from the other side.

## 6. Anti-patterns

Recorded against the first prototype (`TFPWACrossProductHelicity.jl`), kept in
the tree as the negative example. It is numerically correct once the energy bug
is fixed — the objection is structural.

**`BoostRestTable()` — a whole phase behind one instruction.** It runs
`build_rest_table!`, a topological sort over the chain filling a nested `Dict`
of rest-frame momenta before any vertex instruction executes. That is TF-PWA's
`cal_chain_boost`: a batch preprocessor, not a step in a frame narrative. The
test to apply: *if a boost is not a line in the program, it is not IDT.*

**Physics in a sidecar while `objs` stay in the lab.** The prototype carried four
parallel representations — `objs` (barely used), a `rest` dict, `axes_z/axes_x`,
and a separate `lab_p4` input, because the program never drove kinematics
through `objs` at all. The single-frame invariant is not "there exists a sidecar"
but "everything you can index is in the same frame". A sidecar whose contents
are in a *different* frame from `objs` breaks it; the axis marker of §2, which is
in the same frame as everything else, does not.

**`CrossProductVertex(core, outs...)` — a loop in a box.** It iterated both
daughters, applied the bias convention, and updated the axes in one call, so the
decomposable steps (which daughter is measured, which is only transported, when
the axes change) were invisible. Split at the physics seam instead.

**Copying TF-PWA's two-phase architecture and relabelling it.** The root cause of
all three. `cal_chain_boost` then `cal_helicity_angle`, with IDT names pasted on,
produces a facade: correct numbers, no pedagogy. The question to ask is not
"how do I call TF-PWA's algorithm from an instruction?" but "what does this decay
chain look like as a frame narrative?"

**A fifth, added here.** Do not test only against the framework you are porting
*from*. The first prototype validated against its own TF-PWA port, so a bug
shared by both went unnoticed and got written up as a physics finding. §3 of the
validation exists for this reason: it checks the walk against plain IDT
instructions, which have no code in common with either.

### What would count as another failed attempt

- Any instruction whose implementation loops over more than one vertex.
- Any state that `objs` does not carry, or that is in a different frame than `objs`.
- A program whose lines you cannot map one-to-one onto boosts and measurements
  in the decay walk.
- Numerical agreement offered as evidence of faithfulness. It is necessary and
  not close to sufficient — the first prototype had it.
