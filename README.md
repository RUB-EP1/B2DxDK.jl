# B2DxDK Decay Model Analysis

This repository contains a Julia implementation of the analysis for the three-body decay B+ → D- D*+ K+ using Pluto.jl notebooks.
The project implements the amplitude model for studying this decay channel,
including various resonance contributions and interference effects.

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

## Project Structure

```
B2DxDK/
├── data/                         # Production inputs for scripts/
│   ├── final_params_full.json
│   ├── b-decay-events.arrow
│   ├── crosscheck.arrow
│   └── crosscheck_amplitudes_reference.txt
├── archive/
│   ├── angles/                   # Angular convention cross-checks (see README there)
│   ├── data/                     # Historical / auxiliary datasets (incl. crosscheck_event.json)
│   ├── notebooks/                # Saved fit-fraction reference and plot outputs
│   ├── investigation/            # TF-PWA Analysis, ExecutionFlow, tf-pwa submodule
│   ├── flat4b/                   # Flat 4-body phase-space TF-PWA cross-check
│   └── threebodydecays/          # Earlier ThreeBodyDecays.jl model attempts
├── scripts/
│   ├── all_resonances_model.jl                # TF-PWA-aligned CascadeDecays model definitions
│   ├── all_resonances_amplitude_crosscheck.jl # 100-event amplitude regression
│   └── all_resonances_fit_fractions.jl        # Full-sample weighted fit fractions
└── README.md
```

Key files under `data/` (used by `scripts/`):

- `final_params_full.json` — fitted couplings for the production CascadeDecays model
- `b-decay-events.arrow` — full weighted event sample
- `crosscheck.arrow` — 100-event amplitude regression subset
- `crosscheck_amplitudes_reference.txt` — reference amplitudes for regression

Additional historical datasets (including `crosscheck_event.json` for angular checks) are under `archive/data/`. Angular cross-check scripts are under [`archive/angles/`](archive/angles/README.md).

## Installation and Usage

### Prerequisites
- Julia 1.10. The package manager will have to resolve the dependencies for any julia version rather than 1.11.5.
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
[`archive/threebodydecays/`](archive/threebodydecays/README.md).

1. **Install Pluto.jl** (only if opening archived notebooks):
   ```julia
   julia> ] add Pluto
   julia> using Pluto; Pluto.run()
   ```

2. **Run cascade amplitude regression** (recommended entry point):
   ```bash
   julia --project=. scripts/all_resonances_amplitude_crosscheck.jl
   ```

### TF-PWA investigation (archived)

Python/TF-PWA notebooks, execution-flow notes, the `tf-pwa` submodule, and environment
setup scripts are under [`archive/investigation/`](archive/investigation/README.md).
That material supported cross-checks while building the production CascadeDecays workflow;
it is not required to run the Julia regression scripts below.

### All-resonance model and fit fractions (Julia, TF-PWA aligned)

`scripts/all_resonances_model.jl` defines the TF-PWA-aligned amplitude model using
[CascadeDecays.jl](https://github.com/RUB-EP1/CascadeDecays.jl) v0.1.0 for **cascade**
phase space ( $D^*$ at nominal mass).

- `scripts/all_resonances_amplitude_crosscheck.jl` — 100-event amplitude regression on
  `data/crosscheck.arrow` (compared to `data/crosscheck_amplitudes_reference.txt`)
- `scripts/all_resonances_fit_fractions.jl` — full-sample weighted fit fractions on
  `data/b-decay-events.arrow`; compares to `archive/notebooks/all_resonances_fit_fractions.txt`
  (regenerates gitignored `scripts/all_resonances_fit_fractions.txt` when run)

Flat **4-body** phase-space cross-checks (historical; exposed TF-PWA running-mass conventions)
are archived under [`archive/flat4b/`](archive/flat4b/README.md).

Earlier **ThreeBodyDecays.jl** model attempts are archived under
[`archive/threebodydecays/`](archive/threebodydecays/README.md).

TF-PWA investigation material is archived under
[`archive/investigation/`](archive/investigation/README.md).

Run the cascade amplitude regression from the project root:

```bash
julia --project=. scripts/all_resonances_amplitude_crosscheck.jl
```

Run full-sample fit fractions:

```bash
julia --project=. scripts/all_resonances_fit_fractions.jl
```

#### `root_remove_particle2_phase` (X1(2900) only)

Only the three `X1(2900)` branches in `scripts/all_resonances_model.jl`
set `root_remove_particle2_phase=true`. This is **not** interchangeable with the
overall sign encoded by a `_neg` lineshape suffix (see `lineshape_spec`).

- **What it is:** a static Jacob-Wick particle-2 helicity sign,
  $(-1)^{(j_2-\lambda_2)/2}$ when $(j_2-\lambda_2)/2$ is odd, applied at the
  $B^+\to X_1(2900)+({\rm D},K)$ production vertex. It depends only on the spin
  and helicity quantum numbers of the second daughter line (here the spin-1
  $({\rm D},K)$ subsystem), not on event angles or momenta.
- **What it does in code:** `CascadeDecays` always applies this factor once in
  `_vertex_coupling_value`. With `BuggyParticleTwoPhaseLS` on the root vertex,
  the same factor is applied a second time inside the recoupling, so they cancel
  and the net amplitude matches TF-PWA, which does not include this factor at
  that vertex.
- **Why it is not an overall sign:** an overall sign is one factor $\pm1$ on the
  whole chain, independent of helicity. The particle-2 phase flips sign between
  internal helicity components (e.g. $\lambda_2=+1$ vs $\lambda_2=-1$ for
  $j_2=1$). Replacing `root_remove_particle2_phase=true` by a global `_neg` on
  the lineshape would change the relative phases of helicity contributions and
  would not reproduce the TF-PWA amplitude.

## Reference files for the isolated `Psi(4040)` amplitude

Archived TF-PWA and Julia probe material for the isolated `Psi(4040)` test point lives under
[`archive/investigation/`](archive/investigation/README.md) and
[`archive/threebodydecays/notebooks/`](archive/threebodydecays/README.md).

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
