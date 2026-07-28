# scripts

## Helicity-angle cross-checks (TF-PWA vs CascadeDecays)

TF-PWA computes helicity angles by pure boosts plus explicit cross-product axis
transport; `CascadeDecays` realigns the frame at every step. These turn out to be
the same convention — `ToHelicityFrame` is `ToRestFrame` followed by an alignment
rotation, and the cross-product method is what you get by storing that rotation
as `(ẑ, x̂)` instead of applying it.

| file | what it is |
|---|---|
| `CrossProductWalk.jl` | the algorithm as an InstructionalDecayTrees program — four instructions, axes carried inside `objs`, no sidecar |
| `run_crossproduct_walk.jl` | single-event validation: frame trace, TF-PWA equivalence, IDT equivalence, CascadeDecays, frame algebra |
| `run_crossproduct_scan.jl` | the same comparison over N random phase-space points (default 1000) |
| `compare_angles_one_event.jl` | tabulates CascadeDecays against a standalone TF-PWA port, through `KinematicPoint` |
| `TFPWACrossProductHelicity.jl` | direct TF-PWA-shaped port; the reference implementation *and* the negative example of the design note |
| `DESIGN_crossproduct_idt.md` | design note — instruction set, why it is IDT-faithful, anti-patterns |

`CrossProductWalk.jl` is a **temporary local copy**. The instruction set is being
upstreamed to `InstructionalDecayTrees.jl`; once that lands, this file should be
deleted and the scripts should `using InstructionalDecayTrees` instead.

### Running

Most scripts here run with plain `julia --project=. scripts/<name>.jl`.

The cross-product ones additionally need `InstructionalDecayTrees`, which sits in
the root `[extras]` rather than `[deps]`. Stack a throwaway environment carrying
the repo's own `Manifest.toml`, so the same package version is used:

```bash
TENV=$(mktemp -d); cp Manifest.toml "$TENV/"
printf '[deps]\nInstructionalDecayTrees = "1d606af4-d0f8-4ff7-bc8d-eb3f657b7647"\n' > "$TENV/Project.toml"
JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/run_crossproduct_walk.jl
JULIA_LOAD_PATH="@:$TENV:@stdlib" julia --project=. scripts/run_crossproduct_scan.jl 10000
```

`scripts/all_resonances_fit_fractions.jl` needs `Arrow` the same way — same trick,
substituting Arrow's UUID.

## Other

- `all_resonances_fit_fractions.jl` — fit fractions for the full resonance model
- `all_components_independent_amplitude_flow_random_event_test.ipynb` — independent
  Python reimplementation of the TF-PWA amplitude flow, used as an external check
