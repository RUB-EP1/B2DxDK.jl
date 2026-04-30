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
├── notebooks/
│   └── completion.jl          # Main Pluto.jl notebook
├── data/
│   ├── interference_paper.json    # Paper results for comparison
│   ├── interference_tf.json       # TensorFlow results
│   ├── paper_couplings.json      # Coupling parameters
│   ├── backup_400001.json        # Precomputed integrals
│   ├── crosscheck_event.json     # Single event used for angular cross-checks
│   └── ...                       # Additional data files
├── scripts/
│   ├── cal_pw_fraction.py        # Python script for partial wave analysis
│   └── angles/                   # Julia scripts to cross-check angular conventions
└── README.md
```

The `scripts/angles` folder contains small Julia programs (e.g. `explicit.jl`, `with_LDA.jl`) that compute decay angles and cross-check the angular conventions used in the analysis.  
The file `data/crosscheck_event.json` provides a representative event whose four-vectors and derived angles are used as a reference input for these checks.

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

1. **Install Pluto.jl**:
   ```julia
   julia> ] add Pluto
   julia> using Pluto; Pluto.run()
   ```

2. **Open the notebook**:
   - Navigate to the `notebooks/` directory
   - Open `completion.jl` in Pluto

3. **Run the analysis**:
   - The notebook will automatically install required dependencies
   - Execute cells sequentially to perform the analysis


### Using the amplitude extraction

This repository includes a slightly modified version of tf_pwa (https://github.com/jiangyi15/tf-pwa).

Steps to make the analysis code operational:

#### Option A: Conda-based setup (original)
- Conda has to be installed on the system
- Clone this repository
- In console (inside the repo folder):
  - `chmod +x setup_tf_pwa_with_conda.sh`
  - `./setup_tf_pwa_with_conda.sh`

#### Option B: venv-based setup (no Conda)

From the project root:

```bash
chmod +x setup_tf_pwa_with_venv.sh
./setup_tf_pwa_with_venv.sh

# install tf_pwa into the virtual environment
source venv/bin/activate
pip install git+https://github.com/jiangyi15/tf-pwa.git
deactivate
```

The current analysis can be found in `Analysis/Amplitude.ipynb`.


## Reference files for the isolated `Psi(4040)` amplitude

The repository also contains a small set of focused reference files for the
isolated `Psi(4040)` complex-amplitude calculation:

- `Analysis/psi4040_independent_amplitude_flow.ipynb`
  - A self-contained Python notebook that reproduces the isolated TF-PWA
    `Psi(4040)` amplitude step by step, using copied/adapted local functions,
    the TF-PWA configuration values, and the hardcoded probe four-vectors.
  - It is intended as the reference execution flow for the `Psi(4040)` test
    point.

- `Analysis/psi4040_python_function_formula_map.md`
  - A function-to-formula map for the isolated Python notebook.
  - It links the main Python calls to the corresponding mathematical
    expressions, so the notebook calculation can be followed analytically.

- `notebooks/cascade_decays_tfpwa_aligned.jl`
  - A Julia implementation of the same isolated `Psi(4040)` amplitude using
    only Julia packages, in particular `CascadeDecays.jl`,
    `ThreeBodyDecays.jl`, `FourVectors.jl`, and `HadronicLineshapes.jl`.
  - It prints the calculation in a Step 1-7 style comparable to the Python
    execution-flow notebook and shows the remaining constant normalization
    mismatch factor between the package-native Julia convention and TF-PWA.
