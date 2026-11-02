# scripts

| | |
|---|---|
| `model.jl` | the amplitude model (TF-PWA fit) with CascadeDecays; the scripts and `test/runtests.jl` load it with `include` |
| [`angular_check/`](angular_check/) | helicity-angle cross-checks, TF-PWA vs CascadeDecays — see its own README |
| `all_resonances_fit_fractions.jl` | fit fractions for the full resonance model |
| `serialize_model_to_json.jl` | writes the amplitude-serialization files `data/b2dxdk_amplitude_model_{Cminus,Cplus}.json` with CascadeDecaysIO; `test/runtests.jl` checks them |
| `all_components_independent_amplitude_flow_random_event_test.ipynb` | independent Python reimplementation of the TF-PWA amplitude flow, used as an external check |

## Packages in `[extras]`

`Arrow` is a regular dependency, so all scripts above run with
`julia --project=. scripts/<script>.jl`.

`InstructionalDecayTrees` is listed in the root `[extras]` rather than `[deps]`, so
scripts that need it directly (`angular_check/`) fail under a plain
`julia --project=.`. Stack a throwaway environment on the load path instead of
editing the repo's `Project.toml`, and copy the repo's `Manifest.toml` into it so
that the version `CascadeDecays` already loads is used — see
[`angular_check/README.md`](angular_check/README.md).
