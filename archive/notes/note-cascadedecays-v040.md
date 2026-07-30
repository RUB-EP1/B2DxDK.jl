# CascadeDecays v0.1.0 → v0.4.0: the particle-2 frame fix

What changes in the $B^+\to D^-D^{*+}K^+$ model when `CascadeDecays` is upgraded
from `v0.1.0` (currently pinned in `Manifest.toml`) to `v0.4.0`, and what it
means for the TF-PWA agreement.

**Short version.** Only the $X\to D^-K^+$ decay angles change, and only in
$\varphi$, by exactly $\pi$. The model amplitude is *not* invariant: the naive
upgrade breaks $X_1(2900)$ by up to 40% event-by-event. Once the model is made
self-consistent again, the whole particle-2 discrepancy against TF-PWA collapses
from a helicity-dependent recoupling workaround to a **single per-chain sign**.
With that one sign absorbed into `MAGIC_SIGNS`, the upgraded model
reproduces the current amplitudes to machine precision and TF-PWA agreement is
restored at $5\times10^{-15}$.

---

## 1. What the upstream fix does

[`RUB-EP1/CascadeDecays.jl#45`](https://github.com/RUB-EP1/CascadeDecays.jl/pull/45)
(commit `dd32415`, tagged `v0.4.0`) rewrites `helicity_angle_program` to descend
through `_helicity_step_instruction`, which is child-position aware:

```julia
is_binary_vertex(topology, vertex_ind) && child_position == 2 ?
    ToHelicityFrameParticle2(indices) :
    ToHelicityFrame(indices)
```

Before the fix, the descent to a vertex always used `ToHelicityFrame`, whatever
the child position. `helicity_frame_path` (external Wigner alignment) was
already child-aware and is **unchanged**.

Effect on our two topologies — every vertex program, dumped under both versions:

| topology | vertex | v0.1.0 → v0.4.0 |
|---|---|---|
| DxD `(((1,2),3),4)` | $B\to XK$, $X\to D^*D$, $D^*\to D^0\pi$ | identical |
| dk `((1,2),(3,4))` | $B\to D^*X$, $D^*\to D^0\pi$ | identical |
| dk `((1,2),(3,4))` | $X\to D^-K^+$ | `ToHelicityFrame((3,4))` → **`ToHelicityFrameParticle2((3,4))`** |

The DxD topology is untouched because the $D^*$ sits in child position 1 at every
vertex, and every child-2 there ($K$, $D^-$, $\pi^+$) is spinless. In the dk
topology the resonance $X$ is child 2 of the root, and it decays — that is the
one place a particle-2 frame is entered.

Numerically, over the 100 cross-check events:

```
max |Δcosθ|                 = 7.8e-16     (unchanged)
max |Δφ − π  (mod 2π)|      = 8.9e-16     (shifted by exactly π)
```

So the new frame differs from the old one by a rotation of $\pi$ **about $z$** —
both frames put $z$ along $\vec p_X$; they differ in the azimuthal reference.
The measured decay angle enters as $e^{i\lambda_X\varphi}d^{\,j_X}_{\lambda_X 0}(\theta)$,
so the fix multiplies each helicity route by

$$e^{i\lambda_X\pi}=(-1)^{\lambda_X}.$$

## 2. Is the amplitude invariant? No — and here is why

The particle-2 convention has two halves:

1. the Jacob–Wick phase $\eta(\lambda_2)=(-1)^{j_2-\lambda_2}$ in the two-body
   coupling, and
2. the $\pi$-rotated frame for particle 2.

Applied together they cancel and observables do not move; applied singly they do
not. **`v0.1.0` had half of it**: `RecouplingLS` carried $\eta$ at every vertex,
while the descent ignored it. That inconsistency is exactly what
`MissingParticleTwoPhaseLS` was introduced to paper over — it cancels $\eta$,
which makes `v0.1.0` self-consistent in the *$\eta$-free* (TF-PWA) convention.

`v0.4.0` supplies the missing half. Keeping the workaround therefore cancels the
coupling phase but leaves the frame phase standing — half a convention again,
and now the wrong half.

Measured on the 100-event cross-check, per chain, `v0.1.0` vs a **naive upgrade**
(source ported to the new API, model table untouched):

| chain | max abs Δ | ratio v0.4.0 / v0.1.0 |
|---|---|---|
| all 15 DxD chains | `0.0e+00` | $+1$ exactly |
| `X0(2900)_L1_d0` | `0.0e+00` | $+1$ exactly |
| `X1(2900)_L0_d1` | `2.97e-01` | $-45.7 \ldots +213.4$ |
| `X1(2900)_L1_d1` | `1.67e-01` | $-1$ exactly |
| `X1(2900)_L2_d1` | `4.33e-01` | $-35.3 \ldots +43.1$ |

$X_0(2900)$ is spin 0 — $\lambda_X=0$, $(-1)^{\lambda_X}=1$ — so it never moves.
The `L=1` row already used `:standard` (there only $\lambda_X=\pm1$ contributes,
where $\eta=+1$), so it picks up a clean overall $-1$. The two rows carrying the
workaround mix $\lambda_X=0$ against $\lambda_X=\pm1$ with the wrong relative
sign, which is the event-dependent damage.

## 3. Consequence for TF-PWA

TF-PWA omits the particle-2 convention entirely. Under `v0.4.0`, plain
`RecouplingLS` therefore differs from TF-PWA by

$$A^{\text{v0.4.0}} = (-1)^{j_2}\;A^{\text{TF-PWA}},$$

$j_2$ being the spin of the child-2 line that is entered. **This is a constant,
not a helicity-dependent phase** — the whole reason `MissingParticleTwoPhaseLS`
existed disappears.

Verified on a bare $B\to D^*X$, $X\to DK$ chain (no lineshapes, no matching
factors), for every DK-resonance spin $j=0,1,2,3$ and every allowed root $L$,
against the $\eta$-free convention:

| $j_X$ | root $(2L,2S)$ | ratio (min … max over events) | $(-1)^{j}$ |
|---|---|---|---|
| 0 | (2,2) | $+1.000000000000$ | $+1$ |
| 1 | (0,0), (2,2), (4,4) | $-1.000000000000$ | $-1$ |
| 2 | (2,2), (4,4), (6,6) | $+1.000000000000$ | $+1$ |
| 3 | (4,4), (6,6), (8,8) | $-1.000000000000$ | $-1$ |

Imaginary part of the ratio is `0.0e+00` throughout.

How firmly can the sign be attributed to the particle-2 convention? Three
independent pieces, and one caveat:

- the *size* is not in doubt. The entire v0.1.0 → v0.4.0 delta on the production
  model is a pure per-chain $\pm1$ with no event dependence — $+1$ on 16 chains,
  $-1$ on the three $X_1(2900)$ chains, each constant to $<4\times10^{-14}$ over
  100 events;
- the *mechanism* is measured in both halves separately: the coupling half is
  $\eta(\lambda_2)$ exactly (`RecouplingLS` vs the $\eta$-free coupling, error
  `0.0e+00`), the frame half is $\varphi\to\varphi+\pi$ exactly (`8.9e-16`), and
  $\eta(\lambda_2)\,(-1)^{\lambda_2}=(-1)^{j_2}$;
- the *spin dependence* follows $(-1)^{j}$ across $j=0,1,2,3$ (table above).

**Caveat.** None of this isolates the sign from the other signs already in
`MAGIC_SIGNS` — five of the thirteen entries are $-1$ and none of them is
attributed to a named convention. What is established is that the upgrade delta is
this sign and nothing else; the total $X_1(2900)$ sign remains, like the others, a
matching result rather than a derivation. That is why it is stored as a magic sign
and not as a separate derived factor.

### Against real TF-PWA numbers

`archive/flat4b/data/crosscheck_4b.arrow` holds genuine TF-PWA amplitudes for
1000 flat-4b events. The DK chains are insensitive to the flat-4b running-mass
issue documented in `archive/flat4b/README.md`, so those two columns are a clean
external reference (the charmonium rows are expected to disagree there and do):

| model | `X0(2900)` max abs Δ | `X1(2900)` max abs Δ |
|---|---|---|
| `v0.1.0` (current production) | `7.4e-15` | `5.5e-15` |
| `v0.4.0` naive upgrade | `7.4e-15` | **`2.3e-01`** (on $|A|_{\max}=5.2\times10^{-1}$) |
| `v0.4.0` + fix below | `7.4e-15` | `5.5e-15` |

## 4. The fix (applied)

Two edits, both in the model description:

1. **`src/resonance_table.jl`** — `root_recoupling` dropped entirely; every row is
   plain `RecouplingLS`. `src/recoupling.jl` (`MissingParticleTwoPhaseLS`,
   `root_recoupling`) deleted.
2. **`src/matching.jl`** — `MAGIC_SIGNS["X1(2900)"]` flipped $+1 \to -1$, with the
   derivation recorded in a comment next to it. $X_0(2900)$ and every DxD chain
   are unaffected.

Result: **every chain reproduces the current `v0.1.0` amplitudes to machine
precision.** Over the 100 cross-check events, 16 of the 19 chains are
bit-identical (`0.0e+00`); the three $X_1(2900)$ chains agree to
$1.7\times10^{-16}$, $4.3\times10^{-17}$, $3.6\times10^{-16}$ — about
$10^{-15}$ relative, i.e. a couple of ulp. They are not bit-identical because the
sign now enters once per chain instead of term-by-term inside the recoupling sum,
and because $\varphi+\pi$ is computed rather than exact. Agreement against TF-PWA
on the 1000 flat-4b events is unchanged line for line. `test/runtests.jl` passes
271/271. The physics model does not move, so fit fractions are unaffected.

`test/particle2_algebra.jl` is **deleted**. It existed to validate
`MissingParticleTwoPhaseLS` and the claim that the mismatch is helicity dependent
and cannot be absorbed into a scalar — both gone. Its remaining content was either
pure `RecouplingLS` algebra (unchanged by `v0.4.0`, and upstream's concern) or a
duplicate of the cross-check in `test/runtests.jl`.

One check from it moved into `test/runtests.jl`: that CascadeDecays really does
enter the particle-2 frame at the dk $(3,4)$ vertex and at no other vertex. The
magic sign is only right as long as that holds, so it should fail loudly rather
than silently in the amplitude.

### API port required by the upgrade

Independent of the physics, `v0.4.0` moved masses and spins off the system object:

| v0.1.0 | v0.4.0 |
|---|---|
| `CascadeSystem(spins, SystemMasses(...))` | *(gone — masses come from the event)* |
| `DecayChain(topology; propagators, vertices)` | `DecayChain(topology, spins; propagators, vertices)` |
| `CascadeDecay(chains, system, ref_topology; ...)` | `CascadeDecay(chains, ref_topology; ...)` |
| `CascadeKinematics` | `DecayChainKinematics` |

`Project.toml` needs `CascadeDecays = "0.4"` and `InstructionalDecayTrees = "0.3"`.

## 5. Reproducing

`v0.4.0` is on the remote (`dd32415`). Set up a sandbox next to the repo:

```bash
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/RUB-EP1/CascadeDecays.jl", rev="v0.4.0")'
```

The comparison scripts used for this note (vertex-program dump, per-chain
amplitude dump, TF-PWA comparison on the flat-4b sample, and the $(-1)^{j_2}$
spin scan) are not committed; they are small and described in full above.

The upgrade itself is applied on branch `upgrade-cascadedecays-v0.4.0`.
