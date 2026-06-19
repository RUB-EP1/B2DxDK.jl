# Archive

Material that supported developing and validating the $B^+ \to D^- D^{*+} K^+$ amplitude model,
but is **not** part of the current production workflow.

Production code and inputs live at the repo root:

| Location | Role |
|----------|------|
| [`scripts/`](../scripts/) | TF-PWA-aligned [CascadeDecays.jl](https://github.com/RUB-EP1/CascadeDecays.jl) model, amplitude regression, fit fractions |
| [`data/`](../data/) | Fitted couplings (`final_params_full.json`), event samples, amplitude cross-check reference |

Start with [`scripts/all_resonances_amplitude_crosscheck.jl`](../scripts/all_resonances_amplitude_crosscheck.jl).
See the [root README](../README.md) for installation and usage.

## Folders

| Folder | What it is | Details |
|--------|------------|---------|
| [`investigation/`](investigation/) | Python/TF-PWA notebooks, configs, execution-flow notes, `tf-pwa` submodule, environment setup | [investigation/README.md](investigation/README.md) |
| [`threebodydecays/`](threebodydecays/) | Earlier Julia model attempts using [ThreeBodyDecays.jl](https://github.com/RUB-EP1/ThreeBodyDecays.jl) (superseded by CascadeDecays) | [threebodydecays/README.md](threebodydecays/README.md) |
| [`flat4b/`](flat4b/) | Flat 4-body phase-space cross-check vs TF-PWA (cascade production uses fixed $D^*$ mass) | [flat4b/README.md](flat4b/README.md) |
| [`angles/`](angles/) | Standalone Julia checks of helicity / decay-angle conventions | [angles/README.md](angles/README.md) |
| [`data/`](data/) | Historical and auxiliary datasets not needed by `scripts/` — interference matrices, older parameter files, `crosscheck_event.json`, fit-fraction CSVs, helper scripts | *(no separate README)* |
| [`notebooks/`](notebooks/) | Saved reference outputs from earlier work — fit-fraction table (`all_resonances_fit_fractions.txt`), comparison plots under `Plots/` | *(no separate README)* |

## How the pieces relate

```text
investigation/     TF-PWA reference implementation & Psi(4040) probes
       │
       ├── threebodydecays/   first Julia reimplementation (ThreeBodyDecays.jl)
       ├── flat4b/            flat 4b phsp exposed running-mass / BW conventions
       └── angles/            angular convention sanity checks
       │
       ▼
scripts/ + data/     production CascadeDecays workflow (repo root)
```

The archive is kept for reproducibility and for understanding design choices (e.g. why
event-dependent breakup masses matter in flat 4-body phase space, or why `BuggyParticleTwoPhaseLS`
exists). None of it is required to run the production regression scripts.
