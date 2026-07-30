# B2DxDK — archived README (detailed)

This file preserves the previous root README: project structure, physics background,
installation notes, and historical workflow documentation.

**Production** lives at the repo root: [`src/`](../src/) (B2DxDK package), [`test/`](../test/) (regression checks), [`scripts/`](../scripts/) (fit-fraction analysis), and [`data/`](../data/) (couplings and event samples). See the [current README](../README.md).

---

## References

- **Paper**: [arXiv:2406.03156](https://arxiv.org/pdf/2406.03156)
- **InspireHEP**: [2794793](https://inspirehep.net/literature/2794793)
- **Internal Documentation**: [TWiki](https://twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/Bm2DstmDpKm)
- **Internal Code**: [GitLab@CERN](https://gitlab.cern.ch/lhcb-b2oc/analyses/b2oc-aman-bu2dstdk-run12/-/issues/1), [GitLab@EP1](https://gitlab.ep1.rub.de/lhcb/b2oc-aman-bu2dstdk-run12)
- **Full TF2 code**: [fork by Alexander](https://github.com/AlexanderKazatsky/B2DxDK/tree/main)

## Overview

The B+ → D- D*+ K+ decay is a complex three-body decay that involves multiple resonance contributions and interference effects.

## Physics Background

The decay B+ → D- D*+ K+ involves several resonance contributions:

### Resonances Included:
- Charmonium states in $D^*D$ system: `EFF(1++)`, `ηc(3945)`, `χc2(3930)`, `hc(4000)`, `χc1(4010)`, `ψ(4040)`, `hc(4300)`
- Tetraquark candidate in $D^*K$ and $DK$ system: `Tcs0(2870)`, `Tcs1(2900)`

## Production inputs

For the current layout see the [root README](../README.md) and
[`archive/README.md`](README.md); the tree that used to be reproduced here went stale
and was removed rather than kept in sync.

Key files under `data/` (used by `scripts/`):

- `final_params_full.json` — fitted couplings for the production CascadeDecays model
- `b-decay-events.arrow` — full weighted event sample
- `crosscheck.arrow` — 100-event amplitude regression subset
- `crosscheck_amplitudes_reference.txt` — reference amplitudes for regression

Additional historical datasets (including `crosscheck_event.json` for angular checks) are under `archive/data/`. The current angular cross-checks live at [`scripts/angular_check/`](../scripts/angular_check/README.md); earlier standalone ones are under [`archive/angles/`](angles/README.md). For a guide to all archived material, see [`archive/README.md`](README.md).

## Installation and Usage

### Prerequisites
- Julia 1.11 (the `julia` compat bound in `Project.toml`). Other versions will make the package manager re-resolve the dependencies.
- Pluto.jl

For testing the setup in terminal from the project folder, you can run:
```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```
Any problems at this step, should be reported in the project issue tracker.

### Run the analysis

The main reproducible workflow is the TF-PWA-aligned CascadeDecays scripts under `scripts/`
(see [All-resonance model and fit fractions](#all-resonance-model-and-fit-fractions-julia-tf-pwa-aligned)
below). Earlier ThreeBodyDecays.jl notebooks are archived under
[`archive/threebodydecays/`](threebodydecays/README.md).

1. **Install Pluto.jl** (only if opening archived notebooks):
   ```julia
   julia> ] add Pluto
   julia> using Pluto; Pluto.run()
   ```

2. **Run tests** (100-event amplitude regression + model checks):
   ```bash
   julia --project=. test/runtests.jl
   ```

### TF-PWA investigation (archived)

Python/TF-PWA notebooks, execution-flow notes, the `tf-pwa` submodule, and environment
setup scripts are under [`archive/investigation/`](investigation/README.md).
That material supported cross-checks while building the production CascadeDecays workflow;
it is not required to run the Julia regression scripts below.

### All-resonance model and fit fractions (Julia, TF-PWA aligned)

The `B2DxDK` package (`src/`) defines the TF-PWA-aligned amplitude model using
[CascadeDecays.jl](https://github.com/RUB-EP1/CascadeDecays.jl) for **cascade**
phase space ( $D^*$ at nominal mass).

- `test/runtests.jl` — 100-event amplitude regression on `data/crosscheck.arrow`
  (compared to `data/crosscheck_amplitudes_reference.txt`) plus model sanity checks
- `scripts/all_resonances_fit_fractions.jl` — full-sample weighted fit fractions on
  `data/b-decay-events.arrow`; compares to `archive/notebooks/all_resonances_fit_fractions.txt`
  (regenerates gitignored `scripts/all_resonances_fit_fractions.txt` when run)

Flat **4-body** phase-space cross-checks (historical; exposed TF-PWA running-mass conventions)
are archived under [`archive/flat4b/`](flat4b/README.md).

Earlier **ThreeBodyDecays.jl** model attempts are archived under
[`archive/threebodydecays/`](threebodydecays/README.md).

TF-PWA investigation material is archived under
[`archive/investigation/`](investigation/README.md).

Run tests from the project root:

```bash
julia --project=. test/runtests.jl
```

Run full-sample fit fractions:

```bash
julia --project=. scripts/all_resonances_fit_fractions.jl
```

#### Particle-2 phase (historical)

This section used to document a `root_missing_particle2_phase` / `root_recoupling`
field on the rows of `src/resonance_table.jl`, which selected a
`MissingParticleTwoPhaseLS` recoupling at the $X_1(2900)$ dk root so the net
amplitude matched TF-PWA.

Both the field and `MissingParticleTwoPhaseLS` were removed when the model moved to
`CascadeDecays v0.4.0`, which applies the Jacob–Wick particle-2 convention in the
helicity-frame descent as well as in the coupling. The discrepancy against TF-PWA
then reduces to a per-chain sign, carried by `MAGIC_SIGNS["X1(2900)"]` in
`src/matching.jl`. See [`notes/note-cascadedecays-v040.md`](notes/note-cascadedecays-v040.md),
and [`notes/note-phase-two.md`](notes/note-phase-two.md) for the partial-wave algebra.

## Reference files for the isolated `Psi(4040)` amplitude

Archived TF-PWA and Julia probe material for the isolated `Psi(4040)` test point lives under
[`archive/investigation/`](investigation/README.md) and
[`archive/threebodydecays/notebooks/`](threebodydecays/README.md).

Key Python references (paths relative to `archive/investigation/`):

- `Analysis/psi4040_independent_amplitude_flow.ipynb`
  - Self-contained Python notebook reproducing the isolated TF-PWA `Psi(4040)` amplitude
    step by step.
- `Analysis/psi4040_python_function_formula_map.md`
  - Function-to-formula map for the isolated Python notebook.
- `ExecutionFlow/README.md`
  - Documented TF-PWA execution flow for the same probe.

Julia counterparts: `archive/investigation/notebooks/tfpwa_model_aligned.jl` (manual TF-PWA alignment) and
`archive/threebodydecays/notebooks/cascade_decays_tfpwa_aligned.jl` (package-based probe).
