import argparse, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def savefig(fig, out_dir: str, name: str):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_model_comparison(results_dir, out_dir):
    rows   = [r for r in load_csv(os.path.join(results_dir, "overall_metrics.csv"))
              if r["split"] == "test"]
    models = [r["model"].replace("_", " ") for r in rows]
    f1s    = [float(r["macro_f1"]) for r in rows]
    sizes  = [float(r["size_MB"])  for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    ax1.bar(models, f1s, color=colors)
    ax1.set_ylabel("Macro-F1")
    ax1.set_ylim(0.86, 0.95)
    ax1.set_title("Test Macro-F1")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        ax1.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(models, sizes, color=colors)
    ax2.set_ylabel("Model size (MB)")
    ax2.set_title("Model Size")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Model Comparison on Test Set")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_model_comparison.png")


def fig_policy_comparison(results_dir, out_dir):
    rows  = load_csv(os.path.join(results_dir, "policy_comparison.csv"))
    clean = [r for r in rows if r["condition"] == "clean"]
    corr  = [r for r in rows if r["condition"] == "corrupted"]
    labels = [r["policy"].replace("_", "\n") for r in clean]
    c_f1   = [float(r["macro_f1"]) for r in clean]
    d_f1   = [float(r["macro_f1"]) for r in corr]

    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = range(len(labels)), 0.35
    ax.bar([i - w / 2 for i in x], c_f1, w, label="Clean",     color="#4c72b0")
    ax.bar([i + w / 2 for i in x], d_f1, w, label="Corrupted", color="#dd8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.78, 0.98)
    ax.set_title("Policy Comparison — Macro-F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, out_dir, "fig_policy_comparison.png")


def fig_robustness(results_dir, out_dir):
    rows        = load_csv(os.path.join(results_dir, "robustness_macro_f1.csv"))
    models      = sorted(set(r["model"] for r in rows))
    corruptions = sorted(set(r["corruption"] for r in rows))
    palette     = {"Agentic_ViT": "#e74c3c", "EfficientNet-B0": "#3498db",
                   "MobileNetV3-Small": "#2ecc71"}

    fig, axes = plt.subplots(1, len(corruptions), figsize=(14, 4), sharey=True)
    for ax, corruption in zip(axes, corruptions):
        for model in models:
            pts = sorted(
                [(int(r["severity"]), float(r["macro_f1"]))
                 for r in rows if r["model"] == model and r["corruption"] == corruption],
                key=lambda x: x[0],
            )
            sevs, f1s = zip(*pts)
            ax.plot(sevs, f1s, marker="o", label=model,
                    color=palette.get(model, "gray"))
        ax.set_title(corruption.capitalize())
        ax.set_xlabel("Severity")
        ax.grid(alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(fontsize=7)
    fig.suptitle("Corruption Robustness")
    fig.tight_layout()
    savefig(fig, out_dir, "fig_robustness.png")


def fig_le2i(results_dir, out_dir):
    rows   = load_csv(os.path.join(results_dir, "le2i_results.csv"))
    cfgs   = [r["configuration"].replace("_", "\n") for r in rows]
    f1s    = [float(r["macro_f1"])     for r in rows]
    fnrs   = [float(r["fall_fnr_pct"]) for r in rows]
    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#3498db", "#2ecc71"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(cfgs, f1s, color=colors)
    a1.set_ylabel("Macro-F1")
    a1.set_ylim(0.80, 0.94)
    a1.set_title("Le2i Macro-F1")
    a1.grid(axis="y", alpha=0.3)
    a2.bar(cfgs, fnrs, color=colors)
    a2.set_ylabel("Fall FNR (%)")
    a2.set_title("Le2i Fall FNR")
    a2.grid(axis="y", alpha=0.3)
    patches = [
        mpatches.Patch(color="#95a5a6", label="Unordered"),
        mpatches.Patch(color="#3498db", label="Chronological"),
        mpatches.Patch(color="#2ecc71", label="Complete Agentic ViT"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    savefig(fig, out_dir, "fig_le2i_temporal_ablation.png")


def fig_ablation(results_dir, out_dir):
    rows     = load_csv(os.path.join(results_dir, "ablation_results.csv"))
    variants = [r["variant"].replace("_", " ") for r in rows]
    f1s      = [float(r["macro_f1_mean"]) for r in rows]
    cpus     = [float(r["cpu_p95_ms"])    for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71" if f == max(f1s) else "#e74c3c" if f == min(f1s) else "#3498db"
              for f in f1s]
    ax1.barh(variants, f1s, color=colors)
    ax1.set_xlabel("Macro-F1")
    ax1.set_xlim(0.85, 0.96)
    ax1.set_title("Ablation: Macro-F1")
    ax1.axvline(f1s[0], color="gray", ls="--", lw=1)

    ax2.barh(variants, cpus, color="#95a5a6")
    ax2.set_xlabel("CPU P95 Latency (ms)")
    ax2.set_title("Ablation: CPU Latency")
    ax2.axvline(cpus[0], color="gray", ls="--", lw=1)
    fig.tight_layout()
    savefig(fig, out_dir, "fig_ablation.png")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    fig_model_comparison(args.results_dir, args.output_dir)
    fig_policy_comparison(args.results_dir, args.output_dir)
    fig_robustness(args.results_dir, args.output_dir)
    fig_le2i(args.results_dir, args.output_dir)
    fig_ablation(args.results_dir, args.output_dir)
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default="figures")
    main(p.parse_args())
