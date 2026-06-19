# TF-PWA investigation archive

Historical Python/TF-PWA investigation material used while reproducing and cross-checking
the $B^+ \to D^- D^{*+} K^+$ amplitude before the production CascadeDecays Julia workflow.

Production Julia code lives at the repo root under `scripts/`. Fitted couplings used by
`scripts/all_resonances_model.jl` are in `archive/data/final_params_full.json`.

## Layout

```
Analysis/              # TF-PWA notebooks, configs, Psi(4040) execution-flow references
ExecutionFlow/         # Step-by-step TF-PWA amplitude documentation
tf-pwa/                # TF-PWA git submodule
notebooks/
  tfpwa_model_aligned.jl
setup_tf_pwa_with_conda.sh
setup_tf_pwa_with_conda.ps1
setup_tf_pwa_with_venv.sh
```

## TF-PWA setup

From the repo root:

```bash
chmod +x archive/investigation/setup_tf_pwa_with_venv.sh
./archive/investigation/setup_tf_pwa_with_venv.sh
```

Conda variant:

```bash
chmod +x archive/investigation/setup_tf_pwa_with_conda.sh
./archive/investigation/setup_tf_pwa_with_conda.sh
```

The venv is created next to these scripts (`archive/investigation/venv`). The repo-root
`venv/` and `.venv-tfpwa/` directories (gitignored) may also exist from earlier runs.

## Key entry points

- `Analysis/Amplitude.ipynb` — original TF-PWA amplitude notebook
- `Analysis/psi4040_independent_amplitude_flow.ipynb` — isolated Psi(4040) probe
- `ExecutionFlow/README.md` — documented execution flow for the Psi(4040) test point
