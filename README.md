# B2DxDK

Julia package implementing the amplitude model for the three-body decay

$$B^+ \to D^- D^{*+} K^+$$

from the LHCb publication
[Observation of New Charmonium or Charmoniumlike States](https://inspirehep.net/literature/2794793)
([InspireHEP:2794793](https://inspirehep.net/literature/2794793)).

The model is built with [CascadeDecays.jl](https://github.com/RUB-EP1/CascadeDecays.jl)
and aligned with the [TF-PWA](https://github.com/reutera/TF-PWA) reference implementation.

## Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### A. Build the model

```julia
using B2DxDK

cascade = build_all_resonance_cascade()
# optional: build_resonance_cascade("Psi(4040)") for a single resonance
```

Couplings are loaded from `data/final_params_full.json` at package load time.

### B. Execute the model on four-vectors

Final-state four-momenta are passed in the order

$$(p_{D^0},\ p_{\pi^+},\ p_{D^-},\ p_{K^+})$$

The $D^{*+}$ is reconstructed as $D^0\,\pi^+$ (cascade kinematics with $D^{*+}$ at nominal mass).

```julia
using B2DxDK
using CascadeDecays
using FourVectors

cascade = build_all_resonance_cascade()

pD0     = FourVector(px_D0,  py_D0,  pz_D0;  E=E_D0)
piplus  = FourVector(px_pip, py_pip, pz_pip; E=E_pip)
pDminus = FourVector(px_Dm,  py_Dm,  pz_Dm;  E=E_Dm)
pKplus  = FourVector(px_Kp,  py_Kp,  pz_Kp;  E=E_Kp)

point = KinematicPoint(B2DxDK.kinematic_task, (pD0, piplus, pDminus, pKplus))

# amplitude for one resonance (sum over its sub-chains)
amp_psi4040 = only(amplitude(cascade[resonance_chain_names("Psi(4040)")], point))

# coherent sum over all chains (all resonances)
amp_total = only(amplitude(cascade, point))
```

If four-momenta are stored in a table row (as in `data/crosscheck.arrow`), use `event_point(row)` instead.

### Regression checks

```bash
julia --project=. test/runtests.jl
```

## Analysis references

- **Internal documentation**: [TWiki](https://twiki.cern.ch/twiki/bin/viewauth/LHCbPhysics/Bm2DstmDpKm)
- **Internal code**: [GitLab@CERN](https://gitlab.cern.ch/lhcb-b2oc/analyses/b2oc-aman-bu2dstdk-run12/-/issues/1), [GitLab@EP1](https://gitlab.ep1.rub.de/lhcb/b2oc-aman-bu2dstdk-run12)
- **Full TF2 code**: [fork by Alexander Kazatsky](https://github.com/AlexanderKazatsky/B2DxDK/tree/main)

## Archive

Earlier notebooks, investigation material, and convention notes are under
[`archive/`](archive/README.md).

## Acknowledgements

This model was developed in the context of the **LHCb collaboration** analysis of

$$B^+ \to D^- D^{*+} K^+$$

The amplitude structure follows the [TF-PWA](https://github.com/reutera/TF-PWA) framework used in that analysis.
We especially thank **Yi Juang** for the TF-PWA reference implementation and
for extensive cross-checks while building this Julia port.
