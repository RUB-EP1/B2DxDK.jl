"""
Dalitz Comparison Plotter: 4-Case Evaluation Suite
==================================================
Generates individual and comparative Dalitz plots and error distribution plots
comparing Conversion-Corrected Isolated TF-PWA against CascadeDecays across
the complete 2x2 matrix:
(a) Case (a): PDG parent masses + Mixed lineshapes
(b) Case (b): PDG parent masses + Only nominal lineshapes
(c) Case (c): Event-specific parent masses (sqrt(p1^2+p2^2)) + Mixed lineshapes
(d) Case (d): Event-specific parent masses (sqrt(p1^2+p2^2)) + Only nominal lineshapes
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def load_complex(path, n_events=None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Amplitude file not found: {p}")
    if p.suffix == ".bin":
        raw = np.fromfile(str(p), dtype=np.float64)
        if n_events is not None and len(raw) != 2 * n_events:
            raise ValueError(f"File {p} has {len(raw)} Float64 entries, expected {2 * n_events}")
        return raw[0::2] + 1j * raw[1::2]
    else:
        data = np.loadtxt(str(p))
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"File {p} has shape {data.shape}, expected (N, 2)")
        return data[:, 0] + 1j * data[:, 1]

def compute_rel_diff(a_cd, a_tf):
    return np.abs(a_cd - a_tf) / np.maximum(np.abs(a_tf), 1e-15)

def bin_dalitz(x, y, values, bins=90):
    x_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), bins + 1)
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    val_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    with np.errstate(divide="ignore", invalid="ignore"):
        binned = np.where(count > 0, val_sum / count, np.nan)
    return x_edges, y_edges, binned.T

def print_statistics(name, rel_diff):
    print(f"\n--- Statistics: {name} ---")
    print(f"  Median:  {np.median(rel_diff):.3e}")
    print(f"  Mean:    {np.mean(rel_diff):.3e}")
    print(f"  90%:     {np.percentile(rel_diff, 90):.3e}")
    print(f"  95%:     {np.percentile(rel_diff, 95):.3e}")
    print(f"  99%:     {np.percentile(rel_diff, 99):.3e}")
    print(f"  Maximum: {np.max(rel_diff):.3e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Dalitz comparison plots for all 4 cases.")
    parser.add_argument("--events", type=str, default=None, help="Path to sampled events JSON")
    parser.add_argument("--tfpwa", type=str, default=None, help="Path to TF-PWA amplitude file")
    parser.add_argument("--case-a", "--combo1", dest="case_a", type=str, default=None, help="Path to Case (a) amplitude file")
    parser.add_argument("--case-b", "--combo2", dest="case_b", type=str, default=None, help="Path to Case (b) amplitude file")
    parser.add_argument("--case-c", "--combo4", dest="case_c", type=str, default=None, help="Path to Case (c) amplitude file")
    parser.add_argument("--case-d", "--combo3", dest="case_d", type=str, default=None, help="Path to Case (d) amplitude file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    suite_dir = work_dir.parent
    amp_dir = suite_dir / "amp_data"
    plots_dir = suite_dir / "plots"

    root_candidates = [
        suite_dir.parent,
        suite_dir.parent / "B2DxDK.jl",
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test"),
        Path("c:/Users/gamma/Documents/Playground/Antigravity_Test/B2DxDK.jl"),
    ]

    events_path = None
    if args.events:
        events_path = Path(args.events)
    else:
        for r in root_candidates:
            cand = r / "data" / "sampled_events_tfpwa.json"
            if cand.is_file():
                events_path = cand
                break
        if events_path is None:
            events_path = suite_dir.parent / "data" / "sampled_events_tfpwa.json"

    out_dir = Path(args.output_dir).resolve() if args.output_dir else plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tfpwa_path = Path(args.tfpwa) if args.tfpwa else amp_dir / "isolated_tfpwa_amp.txt"
    ca_path = Path(args.case_a) if args.case_a else amp_dir / "case_a_cd_amp.txt"
    cb_path = Path(args.case_b) if args.case_b else amp_dir / "case_b_cd_amp.txt"
    cc_path = Path(args.case_c) if args.case_c else amp_dir / "case_c_cd_amp.txt"
    cd_path = Path(args.case_d) if args.case_d else amp_dir / "case_d_cd_amp.txt"

    print(f"Loading kinematic coordinates from: {events_path}")
    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p4 = data["p4"]
    n_events = len(p4["D"])

    # Dalitz coordinates: m^2(D* D) and m^2(D K)
    pD = np.asarray(p4["D"])
    pK = np.asarray(p4["K"])
    pD0 = np.asarray(p4["D0"])
    ppi = np.asarray(p4["pi"])
    pDst = pD0 + ppi

    p_dstd = pDst + pD
    m2_dstd = p_dstd[:, 0] ** 2 - np.sum(p_dstd[:, 1:] ** 2, axis=-1)

    p_dk = pD + pK
    m2_dk = p_dk[:, 0] ** 2 - np.sum(p_dk[:, 1:] ** 2, axis=-1)

    print(f"Loading amplitudes for {n_events} events...")
    a_tf = load_complex(tfpwa_path, n_events)
    a_a  = load_complex(ca_path, n_events)
    a_b  = load_complex(cb_path, n_events)
    a_c  = load_complex(cc_path, n_events)
    a_d  = load_complex(cd_path, n_events)

    diff_a = compute_rel_diff(a_a, a_tf)
    diff_b = compute_rel_diff(a_b, a_tf)
    diff_c = compute_rel_diff(a_c, a_tf)
    diff_d = compute_rel_diff(a_d, a_tf)

    print_statistics("Case (a): PDG Parent + Mixed Lineshapes", diff_a)
    print_statistics("Case (b): PDG Parent + Nominal Lineshapes", diff_b)
    print_statistics("Case (c): Event-spec Parent + Mixed Lineshapes", diff_c)
    print_statistics("Case (d): Event-spec Parent + Nominal Lineshapes", diff_d)

    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    cases = [
        ("(a) PDG Parent + Mixed Lineshapes vs TF-PWA (Max: {:.2e})".format(np.max(diff_a)), diff_a, "dalitz_case_a_pdg_mixed.png", (1e-10, 1e-4)),
        ("(b) PDG Parent + Nominal Lineshapes vs TF-PWA (Max: {:.2e})".format(np.max(diff_b)), diff_b, "dalitz_case_b_pdg_nominal.png", (1e-10, 1e-4)),
        ("(c) Event-Spec Parent + Mixed Lineshapes vs TF-PWA (Max: {:.2e})".format(np.max(diff_c)), diff_c, "dalitz_case_c_eventspec_mixed.png", (1e-10, 1e-4)),
        ("(d) Event-Spec Parent + Nominal Lineshapes vs TF-PWA (Max: {:.2e})".format(np.max(diff_d)), diff_d, "dalitz_case_d_eventspec_nominal.png", (1e-10, 1e-4)),
    ]

    # Individual Dalitz plots
    for title, diff, fname, (vmin, vmax) in cases:
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
        xe, ye, binned = bin_dalitz(m2_dstd, m2_dk, diff, bins=90)
        im = ax.pcolormesh(xe, ye, binned, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", shading="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Relative Difference $|A_{\mathrm{CD}} - A_{\mathrm{TF}}| / |A_{\mathrm{TF}}|$")
        ax.set_xlabel(r"$m^2(D^* D)$ [$\mathrm{GeV}^2$]")
        ax.set_ylabel(r"$m^2(DK)$ [$\mathrm{GeV}^2$]")
        ax.set_title(title, pad=10, fontweight="bold")
        fig.tight_layout()
        save_p = out_dir / fname
        fig.savefig(save_p, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_p}")

    # Combined 2x2 Dalitz Plot matching the 2 variation places:
    # Row 1: PDG parent ((a), (b))
    # Row 2: Event-spec parent ((c), (d))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    grid_configs = [
        # (row, col, title, diff, scale)
        (0, 0, "(a) PDG Parent + Mixed LS\n(Max: {:.2e})".format(np.max(diff_a)), diff_a, (1e-10, 1e-4)),
        (0, 1, "(b) PDG Parent + Nominal LS\n(Max: {:.2e})".format(np.max(diff_b)), diff_b, (1e-10, 1e-4)),
        (1, 0, "(c) Event-Spec Parent + Mixed LS\n(Max: {:.2e})".format(np.max(diff_c)), diff_c, (1e-10, 1e-4)),
        (1, 1, "(d) Event-Spec Parent + Nominal LS\n(Max: {:.2e})".format(np.max(diff_d)), diff_d, (1e-10, 1e-4)),
    ]

    for r, c, title, diff, (vmin, vmax) in grid_configs:
        ax = axes[r, c]
        xe, ye, binned = bin_dalitz(m2_dstd, m2_dk, diff, bins=90)
        im = ax.pcolormesh(xe, ye, binned, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", shading="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Relative Difference $|A_{\mathrm{CD}} - A_{\mathrm{TF}}| / |A_{\mathrm{TF}}|$", fontsize=10)
        ax.set_xlabel(r"$m^2(D^* D)$ [$\mathrm{GeV}^2$]")
        if c == 0:
            ax.set_ylabel(r"$m^2(DK)$ [$\mathrm{GeV}^2$]")
        ax.set_title(title, pad=10, fontweight="bold")

    fig.tight_layout()
    combined_p = out_dir / "dalitz_comparison_all_four_cases.png"
    fig.savefig(combined_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2x2 combined plot: {combined_p}")

    # Overlaid 1D Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bins = np.logspace(-11, -4.5, 110)
    colors = ["#fdbf6f", "#e31a1c", "#1f78b4", "#6a3d9a"]
    labels = [
        f"(a) PDG Parent + Mixed LS (Max = {np.max(diff_a):.2e})",
        f"(b) PDG Parent + Nominal LS (Max = {np.max(diff_b):.2e})",
        f"(c) Event-Spec Parent + Mixed LS (Max = {np.max(diff_c):.2e})",
        f"(d) Event-Spec Parent + Nominal LS (Max = {np.max(diff_d):.2e})",
    ]
    ax.hist(diff_a, bins=bins, color=colors[0], alpha=0.55, label=labels[0], edgecolor="none")
    ax.hist(diff_b, bins=bins, color=colors[1], alpha=0.50, label=labels[1], edgecolor="none")
    ax.hist(diff_c, bins=bins, color=colors[2], alpha=0.55, label=labels[2], edgecolor="none")
    ax.hist(diff_d, bins=bins, color=colors[3], alpha=0.50, label=labels[3], edgecolor="none")

    ax.set_xscale("log")
    ax.set_xlabel(r"Pointwise Relative Difference $|A_{\mathrm{CD}} - A_{\mathrm{TF}}| / |A_{\mathrm{TF}}|$")
    ax.set_ylabel("Number of Events / Bin")
    ax.set_title("Relative Difference Distributions Across 50,000 Events (All 4 Cases)", fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    dist_p = out_dir / "distribution_comparison_all_four_cases.png"
    fig.savefig(dist_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot: {dist_p}")

if __name__ == "__main__":
    main()
