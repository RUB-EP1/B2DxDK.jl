# scripts

| | |
|---|---|
| [`angular_check/`](angular_check/) | helicity-angle cross-checks, TF-PWA vs CascadeDecays — see its own README |
| `all_resonances_fit_fractions.jl` | fit fractions for the full resonance model |
| `all_components_independent_amplitude_flow_random_event_test.ipynb` | independent Python reimplementation of the TF-PWA amplitude flow, used as an external check |

## Packages in `[extras]`

`Arrow` and `InstructionalDecayTrees` are listed in the root `[extras]` rather
than `[deps]`, so scripts that need them fail under a plain `julia --project=.`.
Stack a throwaway environment on the load path instead of editing the repo's
`Project.toml`.

`all_resonances_fit_fractions.jl` needs `Arrow`:

```bash
AENV=$(mktemp -d)
julia --project="$AENV" -e 'using Pkg; Pkg.add(name="Arrow", version="2.8")'
JULIA_LOAD_PATH="@:$AENV:@stdlib" julia --project=. scripts/all_resonances_fit_fractions.jl
```

For `InstructionalDecayTrees` copy the repo's `Manifest.toml` into the temp env
as well, so the same version resolves that `CascadeDecays` already loads — see
[`angular_check/README.md`](angular_check/README.md).
