"""
Unified Execution Orchestrator: 4-Case Suite
============================================
Executes all 4 comparison cases against Conversion-Corrected Isolated TF-PWA:
(a) Case (a): PDG parent masses + Mixed lineshapes
(b) Case (b): PDG parent masses + Only nominal lineshapes
(c) Case (c): Event-specific parent masses (sqrt(p1^2+p2^2)) + Mixed lineshapes
(d) Case (d): Event-specific parent masses (sqrt(p1^2+p2^2)) + Only nominal lineshapes
and produces publication-quality Dalitz plots and histograms.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

def run_cmd(cmd, desc, cwd=None):
    print(f"\n{'=' * 70}\n==> {desc}\n    Command: {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    subprocess.run(cmd, cwd=cwd, check=True)
    elapsed = time.time() - t0
    print(f"--> Done in {elapsed:.2f}s")
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Run complete 4-case Dalitz comparison evaluation suite.")
    parser.add_argument("--events", type=str, default=None, help="Path to sampled events JSON")
    parser.add_argument("--skip-tfpwa", action="store_true", help="Skip isolated TF-PWA evaluation if already run")
    parser.add_argument("--skip-a", "--skip-combo1", dest="skip_a", action="store_true", help="Skip Case (a) if already run")
    parser.add_argument("--skip-b", "--skip-combo2", dest="skip_b", action="store_true", help="Skip Case (b) if already run")
    parser.add_argument("--skip-c", "--skip-combo4", dest="skip_c", action="store_true", help="Skip Case (c) if already run")
    parser.add_argument("--skip-d", "--skip-combo3", dest="skip_d", action="store_true", help="Skip Case (d) if already run")
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    scripts_dir = work_dir / "scripts"
    amp_dir = work_dir / "amp_data"
    plots_dir = work_dir / "plots"
    amp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        work_dir.parent / "data" / "sampled_events_tfpwa.json",
        work_dir.parent / "B2DxDK.jl" / "data" / "sampled_events_tfpwa.json",
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test/data/sampled_events_tfpwa.json"),
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test/B2DxDK.jl/data/sampled_events_tfpwa.json"),
    ]

    events_path = Path(args.events).resolve() if args.events else next((p for p in candidates if p.exists()), None)
    if not events_path:
        raise FileNotFoundError("Could not find sampled_events_tfpwa.json. Pass via --events.")

    julia_project = None
    for cand in [
        Path("c:/Users/gamma/Documents/Playground/B2DxDK.jl_fresh"),
        work_dir.parent / "B2DxDK.jl",
        work_dir.parent,
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test/B2DxDK.jl"),
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test"),
    ]:
        if (cand / "Project.toml").is_file():
            julia_project = cand
            break
    if julia_project is None:
        julia_project = work_dir.parent

    print(f"Target sampled events file: {events_path}")
    print(f"Julia project directory:   {julia_project}")
    py_exe = sys.executable

    tfpwa_txt = amp_dir / "isolated_tfpwa_amp.txt"
    case_a_txt = amp_dir / "case_a_cd_amp.txt"
    case_b_txt = amp_dir / "case_b_cd_amp.txt"
    case_c_txt = amp_dir / "case_c_cd_amp.txt"
    case_d_txt = amp_dir / "case_d_cd_amp.txt"

    # 1. Evaluate Isolated TF-PWA
    if not (args.skip_tfpwa and tfpwa_txt.exists()):
        run_cmd(
            [py_exe, str(scripts_dir / "isolated_tfpwa.py"), "--events", str(events_path), "--output", str(tfpwa_txt)],
            "Evaluating Conversion-Corrected Isolated TF-PWA",
            cwd=work_dir,
        )

    # 2. Evaluate Case (a): PDG Parent + Mixed Lineshapes
    if not (args.skip_a and case_a_txt.exists()):
        run_cmd(
            ["julia", "-t", "auto", f"--project={julia_project}", str(scripts_dir / "compare_case_a_pdg_mixed.jl"), str(events_path), str(case_a_txt)],
            "Evaluating Case (a): PDG Parent Masses + Mixed Lineshapes",
            cwd=work_dir,
        )

    # 3. Evaluate Case (b): PDG Parent + Nominal Lineshapes
    if not (args.skip_b and case_b_txt.exists()):
        run_cmd(
            ["julia", "-t", "auto", f"--project={julia_project}", str(scripts_dir / "compare_case_b_pdg_nominal.jl"), str(events_path), str(case_b_txt)],
            "Evaluating Case (b): PDG Parent Masses + Nominal Lineshapes",
            cwd=work_dir,
        )

    # 4. Evaluate Case (c): Event-Specific Parent + Mixed Lineshapes
    if not (args.skip_c and case_c_txt.exists()):
        run_cmd(
            ["julia", "-t", "auto", f"--project={julia_project}", str(scripts_dir / "compare_case_c_eventspec_mixed.jl"), str(events_path), str(case_c_txt)],
            "Evaluating Case (c): Event-Specific Parent Masses + Mixed Lineshapes",
            cwd=work_dir,
        )

    # 5. Evaluate Case (d): Event-Specific Parent + Nominal Lineshapes
    if not (args.skip_d and case_d_txt.exists()):
        run_cmd(
            ["julia", "-t", "auto", f"--project={julia_project}", str(scripts_dir / "compare_case_d_eventspec_nominal.jl"), str(events_path), str(case_d_txt)],
            "Evaluating Case (d): Event-Specific Parent Masses + Nominal Lineshapes",
            cwd=work_dir,
        )

    # 6. Generate Dalitz plots and distribution comparisons
    run_cmd(
        [
            py_exe, str(scripts_dir / "plot_dalitz_comparisons.py"),
            "--events", str(events_path),
            "--tfpwa", str(tfpwa_txt),
            "--case-a", str(case_a_txt),
            "--case-b", str(case_b_txt),
            "--case-c", str(case_c_txt),
            "--case-d", str(case_d_txt),
            "--output-dir", str(plots_dir),
        ],
        "Generating 4-Case Dalitz and Distribution Comparison Plots",
        cwd=work_dir,
    )

    print("\n" + "=" * 70)
    print("ALL 4 EVALUATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print(f"Results located in: {plots_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
