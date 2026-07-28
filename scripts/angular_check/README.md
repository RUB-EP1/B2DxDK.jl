# Angular cross-checks: TF-PWA vs CascadeDecays

TF-PWA computes helicity angles by pure boosts plus explicit cross-product axis
transport; `CascadeDecays` realigns the frame at every step. These turn out to be
the same convention — `ToHelicityFrame` is `ToRestFrame` followed by an alignment
rotation, and the cross-product method is what you get by storing that rotation
as `(ẑ, x̂)` instead of applying it.

Conclusion: for **vertex angles** the two agree to 4.6e-13 rad worst case over
10k phase-space points, provided `initial_frame = CurrentFrame()`. See
[issue #22](https://github.com/RUB-EP1/B2DxDK.jl/issues/22).

| file | what it is |
|---|---|
| `CrossProductWalk.jl` | the algorithm as an InstructionalDecayTrees program — four instructions, axes carried inside `objs`, no sidecar |
| `run_crossproduct_walk.jl` | single-event validation: frame trace, TF-PWA equivalence, IDT equivalence, CascadeDecays, frame algebra |
| `run_crossproduct_scan.jl` | the same comparison over N random phase-space points (default 1000) |
| `compare_angles_one_event.jl` | tabulates CascadeDecays against a standalone TF-PWA port, through `KinematicPoint` |
| `TFPWACrossProductHelicity.jl` | direct TF-PWA-shaped port; the reference implementation *and* the negative example of the design note |
| `DESIGN_crossproduct_idt.md` | design note — instruction set, why it is IDT-faithful, anti-patterns |

`CrossProductWalk.jl` is a **temporary local copy**. The instruction set is being
upstreamed in [InstructionalDecayTrees.jl#31](https://github.com/RUB-EP1/InstructionalDecayTrees.jl/pull/31);
once that lands, delete this file and `using InstructionalDecayTrees` instead.

## Running

`compare_angles_one_event.jl` runs with plain `julia --project=.`. The
cross-product scripts additionally need `InstructionalDecayTrees`, which sits in
the root `[extras]` rather than `[deps]`. Stack a throwaway environment carrying
the repo's own `Manifest.toml`, so the same package version resolves that
`CascadeDecays` already loads:

```bash
TENV=$(mktemp -d); cp Manifest.toml "$TENV/"
printf '[deps]\nInstructionalDecayTrees = "1d606af4-d0f8-4ff7-bc8d-eb3f657b7647"\n' > "$TENV/Project.toml"
JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/angular_check/run_crossproduct_walk.jl
JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/angular_check/run_crossproduct_scan.jl 10000
```

## Scope

Vertex angles only. The external Wigner alignment path (`helicity_frame_path`,
`wigner_finals`) — where `ToHelicityFrameParticle2` is actually used, and where a
genuine difference could still live — is not covered here, nor is anything at the
amplitude level.
