"""
MetricReporter  —  Step 33
----------------------------
Formats and saves the full metric report with:
  - per-category tables
  - radar summary chart
  - time-series plots for key signals
  - JSON export for downstream analysis
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricReporter:

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_json(self, metrics: dict,
                   task_name: str = "task",
                   filename : str = None) -> str:

        fname = filename or f"metrics_{task_name}.json"
        path  = os.path.join(self.results_dir, fname)

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, float) and (
                    np.isnan(obj) or np.isinf(obj)):
                return None
            return obj

        with open(path, "w") as f:
            json.dump(_clean(metrics), f, indent=2)
        return path

    def print_report(self, metrics: dict,
                      task_name: str = "task") -> None:

        cats = ["behavioral","learning","reasoning","neural",
                "meta_learning","neuromodulation","energy"]
        labels = {
            "behavioral"    : "Behavioral",
            "learning"      : "Learning",
            "reasoning"     : "Reasoning",
            "neural"        : "Neural",
            "meta_learning" : "Meta-learning",
            "neuromodulation": "Neuromodulation",
            "energy"        : "Energy",
        }

        print(f"\n{'='*60}")
        print(f"  Phase 13 — Step 33: Metric Report  [{task_name}]")
        print(f"{'='*60}")

        for cat in cats:
            cat_metrics = metrics.get(cat, {})
            if not cat_metrics:
                continue
            print(f"\n  {labels[cat]}:")
            for metric, value in cat_metrics.items():
                if value is None:
                    vs = "n/a"
                elif isinstance(value, float):
                    vs = f"{value:.4f}"
                else:
                    vs = str(value)
                bar_val = (value if isinstance(value, float)
                           and 0.0 <= value <= 1.0 else None)
                bar = ""
                if bar_val is not None:
                    bar = "  " + "#" * int(bar_val * 20)
                print(f"    {metric:<30s}: {vs:>10s}{bar}")

        print(f"\n  steps recorded: {metrics.get('step_count', 0):,}")
        print(f"{'='*60}\n")

    def plot_summary(self, metrics: dict,
                      task_name: str = "task",
                      filename : str = None) -> str:
        """
        Bar chart of all normalised scores per category.
        """
        cats = ["behavioral","learning","reasoning","neural",
                "meta_learning","neuromodulation","energy"]
        cat_means = {}
        for cat in cats:
            vals = [v for v in metrics.get(cat, {}).values()
                    if isinstance(v, float)
                    and not (np.isnan(v) or np.isinf(v))
                    and 0.0 <= v <= 1.0]
            cat_means[cat] = float(np.mean(vals)) if vals else 0.0

        labels = [c.replace("_"," ") for c in cats]
        values = [cat_means[c] for c in cats]
        colors = ["#534AB7","#1D9E75","#D85A30",
                  "#BA7517","#7F77DD","#5DCAA5","#EF9F27"]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(labels, values, color=colors, alpha=0.85,
                      width=0.6)
        ax.axhline(np.mean(values), color="gray", ls="--",
                   lw=1, label=f"mean={np.mean(values):.3f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean normalised score")
        ax.set_title(
            f"Phase 13 — Step 33 metric summary: {task_name}")
        ax.legend(fontsize=9)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        fname = filename or f"metric_summary_{task_name}.png"
        path  = os.path.join(self.results_dir, fname)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def plot_time_series(self, step_log: list,
                          task_name   : str  = "task",
                          dt          : float = 0.1e-3,
                          filename    : str  = None) -> str:
        """
        Time-series plots for key signals across all categories.
        """
        if not step_log:
            return ""

        fields = ["reward","U","delta_total","alpha_t",
                  "energy_pJ","expl_conf","DA","5HT","NE"]
        labels_map = {
            "reward"     : "Reward",
            "U"          : "Uncertainty Ut",
            "delta_total": "TD error delta_total",
            "alpha_t"    : "Meta-DA alpha_t",
            "energy_pJ"  : "Energy pJ/step",
            "expl_conf"  : "Explanation conf.",
            "DA"         : "Dopamine DA",
            "5HT"        : "Serotonin 5-HT",
            "NE"         : "Norepinephrine NE",
        }
        colors = ["#1D9E75","#534AB7","#D85A30","#EF9F27",
                  "#BA7517","#0F6E56","#185FA5","#639922","#854F0B"]

        available = [f for f in fields if f in step_log[0]]
        n_plots   = len(available)
        if n_plots == 0:
            return ""

        cols  = 3
        rows  = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(15, rows * 3.5))
        axes_flat = axes.flatten() if n_plots > 1 else [axes]

        t_ms   = np.arange(len(step_log)) * dt * 1000
        smooth = lambda v: np.convolve(
            v, np.ones(100)/100, mode="same")

        for ax, field, color in zip(axes_flat, available, colors):
            vals = [s.get(field, 0.0) for s in step_log]
            ax.plot(t_ms, smooth(vals), color=color, lw=1.2)
            ax.set_title(labels_map.get(field, field), fontsize=10)
            ax.set_xlabel("time (ms)", fontsize=8)
            ax.axhline(0, color="gray", ls="--", lw=0.4)

        for ax in axes_flat[len(available):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Phase 13 time-series: {task_name}", fontsize=11)
        plt.tight_layout()
        fname = filename or f"timeseries_{task_name}.png"
        path  = os.path.join(self.results_dir, fname)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path