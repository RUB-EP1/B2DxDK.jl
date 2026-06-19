# Angular convention cross-checks

Small Julia programs used to **cross-check the angular conventions** of the
$B^+ \to D^- D^{*+} K^+$ analysis. They take four-vectors from
`archive/data/crosscheck_event.json` and compute decay angles that enter the amplitude model.

This folder has its **own Julia environment** (`Project.toml`, `Manifest.toml`) because it
depends on geometry utilities not needed by the production `scripts/` workflow.

## Scripts

1. `explicit.jl` — decay angles via explicit Lorentz and spatial rotations
2. `with_LDA.jl` — same angles via `LazyDecayAngles.jl` for independent comparison

## Setup

From the repo root:

```julia
(pkg)> activate archive/angles
(archive/angles) pkg> instantiate
```

Run from the repo root, e.g.:

```bash
julia --project=archive/angles archive/angles/explicit.jl
```
