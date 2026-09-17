# 4-Case Dalitz Comparison Suite: CascadeDecays vs TF-PWA

This directory contains a clean, reproducible evaluation suite comparing **float64-conversion-corrected isolated TF-PWA** against **CascadeDecays** across the complete $2 \times 2$ matrix of variations on a 50,000-event phase-space sample for $B^+ \to D^{*+} D^- K^+ \to (D^0 \pi^+) D^- K^+$.

Both charge configurations supported in the experimental TF-PWA model are implemented and evaluated:
- **$C_-$ ($c = -1.0$)**: Default negative charge configuration.
- **$C_+$ ($c = +1.0$)**: Alternative positive charge configuration.

---

## Decay Channels: $C_-$ and $C_+$

- **$C_-$ ($c = -1.0$)**: **$B^+ \to D^{*+} D^- K^+$**
  - **$DK$ channel ($D^- K^+$)**: Forms the open-charm exotic tetraquark states **$X_0(2900)$** and **$X_1(2900)$**, which are present in this decay.
  - This is the default configuration in the model (`config_a.yml: extra_var: c: default: -1`).

- **$C_+$ ($c = +1.0$)**: **$B^+ \to D^{*-} D^+ K^+$**
  - **$DK$ channel ($D^+ K^+$)**: Carries doubly positive charge $(+2)$ and cannot form $X(2900)$ resonances; therefore, **$X_0(2900)$** and **$X_1(2900)$** are absent in this decay.

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

### $C_-$ Configuration ($c = -1.0$)
| Case | Script | Place 1: Lineshape | Place 2: Parent Mass | Max Rel. Diff. | Median Rel. Diff. | Key Characteristic |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **Case (a)** | `compare_case_a_pdg_mixed.jl` | Mix (nominal + event-spec) | PDG masses | **$3.50 \times 10^{-5}$** | $9.81 \times 10^{-9}$ | Isolates parent PDG mass shift. |
| **Case (b)** | `compare_case_b_pdg_nominal.jl` | Only nominal masses | PDG masses | **$3.50 \times 10^{-5}$** | $1.03 \times 10^{-8}$ | Purely nominal lineshapes and parents. |
| **Case (c)** | `compare_case_c_eventspec_mixed.jl` | Mix (nominal + event-spec) | $\sqrt{(p_1+p_2)^2}$ | **$7.55 \times 10^{-8}$** | $1.96 \times 10^{-10}$ | **Machine-precision floor** across entire Dalitz plot. |
| **Case (d)** | `compare_case_d_eventspec_nominal.jl` | Only nominal masses | $\sqrt{(p_1+p_2)^2}$ | **$3.02 \times 10^{-7}$** | $4.06 \times 10^{-10}$ | Isolates lineshape effect ($10^{-7}$ threshold band). |

### $C_+$ Configuration ($c = +1.0$)
| Case | Script | Place 1: Lineshape | Place 2: Parent Mass | Max Rel. Diff. | Median Rel. Diff. | Key Characteristic |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **Case (a)** | `compare_case_a_pdg_mixed.jl` | Mix (nominal + event-spec) | PDG masses | **$1.37 \times 10^{-5}$** | $5.87 \times 10^{-9}$ | Isolates parent PDG mass shift ($X_{0,1}(2900) = 0$). |
| **Case (b)** | `compare_case_b_pdg_nominal.jl` | Only nominal masses | PDG masses | **$1.37 \times 10^{-5}$** | $6.37 \times 10^{-9}$ | Purely nominal lineshapes and parents. |
| **Case (c)** | `compare_case_c_eventspec_mixed.jl` | Mix (nominal + event-spec) | $\sqrt{(p_1+p_2)^2}$ | **$2.17 \times 10^{-8}$** | $1.64 \times 10^{-10}$ | **Machine-precision floor** across entire Dalitz plot. |
| **Case (d)** | `compare_case_d_eventspec_nominal.jl` | Only nominal masses | $\sqrt{(p_1+p_2)^2}$ | **$3.29 \times 10^{-7}$** | $4.59 \times 10^{-10}$ | Isolates lineshape effect ($10^{-7}$ threshold band). |

---

## Directory Structure

```text
TF-PWA_vs_CascadeDecays/
├── README.md                              # This documentation
├── run_all_Cminus.py                      # Master orchestrator for C- charge configuration
├── run_all_Cplus.py                       # Master orchestrator for C+ charge configuration
├── scripts/                               # Evaluation and plotting scripts
│   ├── isolated_tfpwa.py                  # Standalone conversion-corrected TF-PWA evaluator
│   ├── compare_case_a_pdg_mixed.jl        # Case (a): PDG parent + Mixed lineshapes
│   ├── compare_case_b_pdg_nominal.jl      # Case (b): PDG parent + Only nominal lineshapes
│   ├── compare_case_c_eventspec_mixed.jl  # Case (c): Event-spec parent + Mixed lineshapes
│   ├── compare_case_d_eventspec_nominal.jl# Case (d): Event-spec parent + Only nominal lineshapes
│   └── plot_dalitz_comparisons.py         # 2D Dalitz & 1D distribution plotter
├── amp_data/                              # Human-readable complex amplitude data (.txt)
│   ├── Cminus/                            # C- amplitudes (50,000 events)
│   │   ├── isolated_tfpwa_amp.txt
│   │   ├── case_a_cd_amp.txt
│   │   ├── case_b_cd_amp.txt
│   │   ├── case_c_cd_amp.txt
│   │   └── case_d_cd_amp.txt
│   └── Cplus/                             # C+ amplitudes (50,000 events)
│       ├── isolated_tfpwa_amp.txt
│       ├── case_a_cd_amp.txt
│       ├── case_b_cd_amp.txt
│       ├── case_c_cd_amp.txt
│       └── case_d_cd_amp.txt
└── plots/                                 # Generated publication-quality figures (.png)
    ├── Cminus/                            # C- Dalitz & distribution plots
    │   ├── dalitz_case_a_pdg_mixed.png
    │   ├── dalitz_case_b_pdg_nominal.png
    │   ├── dalitz_case_c_eventspec_mixed.png
    │   ├── dalitz_case_d_eventspec_nominal.png
    │   ├── dalitz_comparison_all_four_cases.png
    │   └── distribution_comparison_all_four_cases.png
    └── Cplus/                             # C+ Dalitz & distribution plots
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
They can be easily loaded in Python via `np.loadtxt("amp_data/Cminus/case_c_cd_amp.txt")` or in Julia via `readdlm("amp_data/Cminus/case_c_cd_amp.txt", comments=true)`.

---

## Quick Start / Execution

### Master Orchestrators
To run the full suite for either charge configuration:
```bash
# Evaluate C- and generate plots in plots/Cminus/
python run_all_Cminus.py

# Evaluate C+ and generate plots in plots/Cplus/
python run_all_Cplus.py
```

Optional flags for both runners:
- `--skip-tfpwa`: Skip isolated TF-PWA evaluation if output already exists.
- `--skip-a`, `--skip-b`, `--skip-c`, `--skip-d`: Skip respective Julia evaluations if outputs already exist.

### Run Individual Scripts
From the `TF-PWA_vs_CascadeDecays/` directory:

- **Isolated TF-PWA**:
  ```bash
  python scripts/isolated_tfpwa.py --charge Cplus
  ```
- **Case (c) in Julia**:
  ```bash
  julia -t auto --project=.. scripts/compare_case_c_eventspec_mixed.jl --charge=Cplus
  ```
- **Plotting**:
  ```bash
  python scripts/plot_dalitz_comparisons.py --charge Cplus
  ```
