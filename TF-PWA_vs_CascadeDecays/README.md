# 4-Case Dalitz Comparison Suite: CascadeDecays vs TF-PWA

This directory contains a clean, reproducible evaluation suite comparing **float64-conversion-corrected Isolated TF-PWA** against **CascadeDecays** across the complete $2 \times 2$ matrix of variations on a 50,000-event phase-space sample for $B^+ \to D^{*+} D^- K^+ \to (D^0 \pi^+) D^- K^+$.

---

## The $2 \times 2$ Parameter Matrix

Two variation places are systematically examined, each with two options:

1. **Place 1: Lineshapes / Breakup Momenta**:
   - **Only Nominal Masses**: Lineshape propagators (shortened to LS in plots), daughter masses, and running breakup momenta use fixed nominal PDG values ($m_{D^*}^{\text{PDG}} = 2.01026\ \text{GeV}, m_D^{\text{PDG}} = 1.86965\ \text{GeV}, m_K^{\text{PDG}} = 0.493677\ \text{GeV}$).
   - **Mix of Nominal & Event-Specific Masses**: Resonance pole masses remain nominal ($m_0$), but decay daughter masses and running breakup momenta use the event-specific reconstructed masses from 4-momenta ($m(P_{D^*}), m(p_D), m(p_K)$).

2. **Place 2: Parent Particle Masses**:
   - **PDG Masses**: Parent particles ($B^+, D^*$) in kinematics, system definitions, and vertex barrier factors use fixed nominal PDG values ($m_{B^+}^{\text{PDG}} = 5.27934\ \text{GeV}, m_{D^*}^{\text{PDG}} = 2.01026\ \text{GeV}$).
   - **$\sqrt{(p_1+p_2)^2}$ (Event-Specific)**: Parent particles use the invariant masses reconstructed directly from daughter four-momenta ($\text{mass}(P_B) = \sqrt{(\sum p_i)^2}, \text{mass}(P_{D^*}) = \sqrt{(p_{D^0} + p_{\pi^+})^2}$).

---

## Summary of the Four Cases (50,000 Events)

| Case | Script | Place 1: Lineshape | Place 2: Parent Mass | Max Rel. Diff. | Mean Rel. Diff. | Key Characteristic |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **Case (a)** | `compare_case_a_pdg_mixed.jl` | Mix (nominal + event-spec) | PDG masses | **$3.50 \times 10^{-5}$** | $4.75 \times 10^{-8}$ | Isolates parent PDG mass shift while lineshape matches TF-PWA running widths. |
| **Case (b)** | `compare_case_b_pdg_nominal.jl` | Only nominal masses | PDG masses | **$3.50 \times 10^{-5}$** | $4.80 \times 10^{-8}$ | Purely nominal across both lineshapes and parent masses. |
| **Case (c)** | `compare_case_c_eventspec_mixed.jl` | Mix (nominal + event-spec) | $\sqrt{(p_1+p_2)^2}$ | **$7.55 \times 10^{-8}$** | $3.41 \times 10^{-10}$ | **Machine-precision floor** across entire Dalitz plot (exact algebraic identity). |
| **Case (d)** | `compare_case_d_eventspec_nominal.jl` | Only nominal masses | $\sqrt{(p_1+p_2)^2}$ | **$3.02 \times 10^{-7}$** | $1.63 \times 10^{-9}$ | Isolates lineshape effect: reveals $10^{-7}$ vertical threshold band at $m^2(D^* D) \sim 15.1\ \text{GeV}^2$. |

---

## Directory Structure

```text
TF-PWA_vs_CascadeDecays/
├── README.md                              # This documentation
├── run_all.py                             # Master orchestrator for all 4 cases
├── scripts/                               # Evaluation and plotting scripts
│   ├── isolated_tfpwa.py                  # Standalone conversion-corrected TF-PWA evaluator
│   ├── compare_case_a_pdg_mixed.jl        # Case (a): PDG parent + Mixed lineshapes
│   ├── compare_case_b_pdg_nominal.jl      # Case (b): PDG parent + Only nominal lineshapes
│   ├── compare_case_c_eventspec_mixed.jl  # Case (c): Event-spec parent + Mixed lineshapes
│   ├── compare_case_d_eventspec_nominal.jl# Case (d): Event-spec parent + Only nominal lineshapes
│   └── plot_dalitz_comparisons.py         # 2D Dalitz & 1D distribution plotter
├── amp_data/                              # Generated complex amplitude data
│   ├── isolated_tfpwa_amp.txt             # Conversion-corrected TF-PWA amplitudes
│   ├── case_a_cd_amp.txt                  # Case (a) CascadeDecays amplitudes
│   ├── case_b_cd_amp.txt                  # Case (b) CascadeDecays amplitudes
│   ├── case_c_cd_amp.txt                  # Case (c) CascadeDecays amplitudes
│   └── case_d_cd_amp.txt                  # Case (d) CascadeDecays amplitudes
└── plots/                                 # Generated relative difference plots
    ├── dalitz_case_a_pdg_mixed.png
    ├── dalitz_case_b_pdg_nominal.png
    ├── dalitz_case_c_eventspec_mixed.png
    ├── dalitz_case_d_eventspec_nominal.png
    ├── dalitz_comparison_all_four_cases.png
    └── distribution_comparison_all_four_cases.png
```

---

## Data Format

All amplitude files in `amp_data/` are human-readable ASCII text files (`.txt`), formatted with header:
```text
# real imag
```
followed by 50,000 lines containing space-separated full-precision (`%.17e`) real and imaginary components:
```text
1.23456789012345678e-01 -9.87654321098765432e-02
```
They can be easily loaded in Python via `np.loadtxt("amp_data/isolated_tfpwa_amp.txt")` or in Julia via `readdlm("amp_data/case_a_cd_amp.txt", comments=true)`.

---

## Quick Start / Execution

### Run All 4 Evaluations and Generate Plots
To evaluate all 4 cases across the 50,000-event phase space sample and generate all plots:

```bash
python run_all.py
```

Optional flags:
- `--skip-tfpwa`: Skip isolated TF-PWA evaluation if output already exists.
- `--skip-a`, `--skip-b`, `--skip-c`, `--skip-d`: Skip respective Julia evaluations if outputs already exist.

### Run Individual Scripts
From the `TF-PWA_vs_CascadeDecays/` directory:

- **Isolated TF-PWA**:
  ```bash
  python scripts/isolated_tfpwa.py
  ```
- **Case (a)**:
  ```bash
  julia -t auto --project=.. scripts/compare_case_a_pdg_mixed.jl
  ```
- **Case (b)**:
  ```bash
  julia -t auto --project=.. scripts/compare_case_b_pdg_nominal.jl
  ```
- **Case (c)**:
  ```bash
  julia -t auto --project=.. scripts/compare_case_c_eventspec_mixed.jl
  ```
- **Case (d)**:
  ```bash
  julia -t auto --project=.. scripts/compare_case_d_eventspec_nominal.jl
  ```
- **Plotting**:
  ```bash
  python scripts/plot_dalitz_comparisons.py
  ```

---

## Generated Output Plots
All figures are saved in the `plots/` subdirectory:
- `plots/dalitz_case_a_pdg_mixed.png`: Dalitz relative difference for Case (a).
- `plots/dalitz_case_b_pdg_nominal.png`: Dalitz relative difference for Case (b).
- `plots/dalitz_case_c_eventspec_mixed.png`: Dalitz relative difference for Case (c) (only numerical round-off remaining).
- `plots/dalitz_case_d_eventspec_nominal.png`: Dalitz relative difference for Case (d).
- `plots/dalitz_comparison_all_four_cases.png`: $2 \times 2$ comparative Dalitz plot directly mapping onto the variation matrix.
- `plots/distribution_comparison_all_four_cases.png`: Overlaid 1D log-scale histograms for all 4 cases.
