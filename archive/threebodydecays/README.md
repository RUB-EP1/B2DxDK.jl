# ThreeBodyDecays.jl archive

Earlier Julia attempts to reproduce the $B^+ \to D^- D^{*+} K^+$ amplitude using
[ThreeBodyDecays.jl](https://github.com/RUB-EP1/ThreeBodyDecays.jl) as the main framework.
These did not reach agreement with TF-PWA at the level needed for production analysis.

Production code now lives at the repo root in `scripts/all_resonances_model.jl`, built on
[CascadeDecays.jl](https://github.com/RUB-EP1/CascadeDecays.jl) with a small
ThreeBodyDecays recoupling workaround (`BuggyParticleTwoPhaseLS`).

## Layout

```
notebooks/
  completion.jl                              # original Pluto analysis notebook
  pure_model.jl                              # ThreeBodyDecays pure-model notebook
  tfpwa_model.jl                             # cascade amplitude via ThreeBodyDecays
  all_angles.jl                              # helicity-angle convention checks
  ThreeBodyDecay_analysis_Gemini.jl          # exploratory amplitude study
  cascade_decays_tfpwa_aligned.jl            # isolated Psi(4040) probe (CascadeDecays + TBD)
  cascade_decays_tfpwa_aligned_random_event.jl
scripts/
  pure_model.jl                              # ThreeBodyDecays script counterpart
  psi4040_check.jl                           # isolated Psi(4040) cross-check
  serialization.jl                           # ThreeBodyDecaysIO custom lineshape helpers
```

Shared inputs are under `archive/data/`; TF-PWA investigation files are under
`archive/investigation/`. Archived scripts use `joinpath(@__DIR__, "..", "..", "..", "archive", "data", ...)`
to reach them from `archive/threebodydecays/`.

## Commands

From repo root:

```bash
julia --project=. archive/threebodydecays/scripts/pure_model.jl
julia --project=. archive/threebodydecays/scripts/psi4040_check.jl
julia --project=. archive/threebodydecays/notebooks/cascade_decays_tfpwa_aligned.jl
```

Open Pluto notebooks under `archive/threebodydecays/notebooks/` as usual.
