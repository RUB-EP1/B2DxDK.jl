# Cross check of kinematic variables

The `scripts/angles` folder contains small Julia programs used to **cross-check the angular conventions** of the B\(^+\) → D\(^-\) D\(*+\) K\(^+\) analysis.  
They take four-vectors (e.g. from `data/crosscheck_event.json`) and compute the decay angles that enter the amplitude model.  

This folder has its **own Julia environment** (see `Project.toml` and `Manifest.toml`) because it depends on specific geometry and kinematics utilities that are not needed by the rest of the project. Keeping these dependencies isolated avoids polluting the main environment and ensures reproducible angular cross-checks.

## Scripts in this folder

1. `explicit.jl` – computes decay angles with rotations implemented manually (explicit Lorentz and spatial rotations).
2. `with_LDA.jl` – computes the same angles using the `LazyDecayAngles` library, providing an independent implementation for comparison.

## Installing the dependencies

Dependencies are prescribed in `Manifest.toml`. Make sure the file is present and that you use (approximately) the same Julia version as was used to create the file.
The project relies on two non-registered packages:
- [FourVectors.jl](https://github.com/mmikhasenko/FourVectors.jl.git)
- [LazyDecayAngles.jl](https://github.com/mmikhasenko/LazyDecayAngles.jl.git)

## Setting up the environment

From the project root in a Julia session, enter the package manager (press `]`) and run:

```julia
(pkg)> activate scripts/angles
(scripts/angles) pkg> instantiate
```

This will activate the `scripts/angles` environment and install all required dependencies.

