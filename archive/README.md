# Archive

Material from developing and validating the $B^+ \to D^- D^{*+} K^+$ amplitude model.
Not needed for the current production workflow.

**Production** lives at the repo root: [`src/`](../src/) (B2DxDK package), [`test/`](../test/) (regression checks), [`scripts/`](../scripts/) (fit-fraction analysis), and [`data/`](../data/) (couplings and event samples). Entry point: [`test/runtests.jl`](../test/runtests.jl). See the [root README](../README.md).

---

## investigation/

Python/TF-PWA notebooks, configs, step-by-step execution-flow notes, the `tf-pwa` submodule, and environment setup scripts. This was the main reference while building the Julia model.

→ [investigation/README.md](investigation/README.md)

## threebodydecays/

Earlier Julia attempts to reproduce the amplitude with [ThreeBodyDecays.jl](https://github.com/RUB-EP1/ThreeBodyDecays.jl). Superseded by the CascadeDecays workflow in `scripts/`.

→ [threebodydecays/README.md](threebodydecays/README.md)

## flat4b/

Cross-check against TF-PWA when all four momenta are sampled from flat 4-body phase space (production uses cascade phase space with $D^*$ at nominal mass).

→ [flat4b/README.md](flat4b/README.md)

## angles/

Small standalone Julia scripts that cross-check helicity and decay-angle conventions. Uses `archive/data/crosscheck_event.json`.

→ [angles/README.md](angles/README.md)

## data/

Historical datasets not used by `scripts/`: interference matrices, older parameter files, fit-fraction CSVs, helper scripts, and the single-event JSON for angular checks.

## notebooks/

Saved outputs from earlier work: reference fit-fraction table and comparison plots (`Plots/`).

---

Rough lineage:

```text
investigation/  →  threebodydecays/, flat4b/, angles/
                          ↓
              scripts/ + data/  (repo root)
```

The archive is kept for reproducibility and context (running-width conventions, recoupling workarounds, and similar). You can ignore it if you only want to run the production regression scripts.
