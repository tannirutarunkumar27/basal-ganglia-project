"""
ValidationReport — generates structured report from all results.
Saves JSON summary and matplotlib plots.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict


class ValidationReport:

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_json(self, results: dict,
                   capability_scores: dict,
                   filename: str = "phase12_validation.json") -> str:

        summary = {
            "capability_scores": capability_scores,
            "task_results"     : {},
        }
        for name, r in results.items():
            if r is None:
                continue
            summary["task_results"][name] = {
                "accuracy"          : r.accuracy,
                "mean_reward"       : r.mean_reward,
                "cumulative_reward" : r.cumulative_reward,
                "convergence_step"  : r.convergence_step,
                "regret"            : r.regret,
                "mean_U"            : r.mean_U,
                "mean_delta_total"  : r.mean_delta_total,
                "mean_alpha_t"      : r.mean_alpha_t,
                "mean_expl_conf"    : r.mean_expl_conf,
                "mean_n_rules"      : r.mean_n_rules,
                "total_energy_nJ"   : r.total_energy_nJ,
                "spike_reduction_pct": r.spike_reduction_pct,
                "elapsed_s"         : r.elapsed_s,
            }

        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def plot_capability_radar(self, scores: dict,
                               filename: str = "capability_radar.png"
                               ) -> str:
        """
        Plots capability scores as a bar chart
        (radar charts require extra libs — bar is cleaner).
        """
        caps = [k for k in scores if k != "overall"]
        vals = [scores[k] for k in caps]
        short_names = {
            "learn_rewards"     : "Learn\nrewards",
            "resolve_conflict"  : "Resolve\nconflict",
            "adapt_volatility"  : "Adapt\nvolatility",
            "reason_uncertainty": "Reason\nunder Ut",
            "stable_selection"  : "Stable\nselection",
            "explain_choices"   : "Explain\nchoices",
        }
        labels = [short_names.get(c, c) for c in caps]
        colors = ["#534AB7","#1D9E75","#D85A30",
                  "#BA7517","#185FA5","#993556"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      width=0.55)
        ax.axhline(scores["overall"], color="gray",
                   ls="--", lw=1, label=f"overall={scores['overall']:.3f}")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("capability score [0–1]")
        ax.set_title("Phase 12 — Step 32 capability scores")
        ax.legend(fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def plot_task_comparison(self, results: dict,
                              filename: str = "task_comparison.png"
                              ) -> str:
        """
        Accuracy and mean reward across all tasks.
        """
        names  = list(results.keys())
        accs   = [results[n].accuracy        for n in names]
        rews   = [results[n].mean_reward      for n in names]
        convs  = [(results[n].convergence_step if
                   results[n].convergence_step >= 0 else 3000)
                  for n in names]
        short  = [n.replace("_"," ")[:18] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        colors = ["#534AB7" if a >= 0.65 else "#D85A30"
                  for a in accs]
        axes[0].barh(short, accs, color=colors, alpha=0.85)
        axes[0].axvline(0.65, color="gray", ls="--", lw=0.8)
        axes[0].set_xlabel("accuracy")
        axes[0].set_title("Task accuracy")
        axes[0].set_xlim(0, 1)

        rew_colors = ["#1D9E75" if r >= 0 else "#D85A30"
                      for r in rews]
        axes[1].barh(short, rews, color=rew_colors, alpha=0.85)
        axes[1].axvline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("mean reward")
        axes[1].set_title("Mean reward per step")

        conv_colors = ["#534AB7" if c < 2000 else "#888780"
                       for c in convs]
        axes[2].barh(short, convs, color=conv_colors, alpha=0.85)
        axes[2].axvline(2000, color="gray", ls="--", lw=0.8,
                         label="slow convergence")
        axes[2].set_xlabel("convergence step")
        axes[2].set_title("Convergence speed (lower = faster)")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def plot_learning_curves(self, results: dict,
                              filename: str = "learning_curves.png"
                              ) -> str:
        """Smoothed reward curves for all tasks."""
        n_tasks = len(results)
        cols    = 3
        rows    = (n_tasks + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(15, rows * 4))
        axes_flat = axes.flatten() if n_tasks > 1 else [axes]

        smooth = lambda v, w=100: np.convolve(
            v, np.ones(w)/w, mode="same")

        for ax, (name, r) in zip(axes_flat, results.items()):
            if not r.step_log:
                ax.set_visible(False)
                continue
            rews = [s["reward"] for s in r.step_log]
            t_ms = np.arange(len(rews)) * 0.1
            ax.plot(t_ms, smooth(rews), color="#534AB7",
                    lw=1.2, label="reward")
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            if r.convergence_step >= 0:
                ax.axvline(r.convergence_step * 0.1,
                           color="#D85A30", ls=":", lw=0.8,
                           label=f"conv@{r.convergence_step}")
            ax.set_title(name.replace("_"," "), fontsize=10)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("reward (smoothed)")
            ax.legend(fontsize=7)

        for ax in axes_flat[len(results):]:
            ax.set_visible(False)

        plt.suptitle("Phase 12 — learning curves per task",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def print_full_report(self, results: dict,
                           scores: dict) -> None:
        print("\n" + "="*65)
        print("  Phase 12 — Step 32: Experimental Validation Report")
        print("="*65)
        print(f"\n  Tasks evaluated: {len(results)}")
        print(f"\n  {'Task':<28s} {'Acc':>6s} {'Rew':>8s} "
              f"{'Conv':>7s} {'U':>6s} {'EC':>6s}")
        print("  " + "-"*63)
        for name, r in results.items():
            conv = (str(r.convergence_step)
                    if r.convergence_step >= 0 else "none")
            print(f"  {name:<28s} {r.accuracy*100:5.1f}%"
                  f" {r.mean_reward:+7.3f}"
                  f" {conv:>7s}"
                  f" {r.mean_U:5.3f}"
                  f" {r.mean_expl_conf:5.3f}")
        print("  " + "-"*63)
        print("\n  Capability scores:")
        for cap, score in scores.items():
            bar = "#" * int(score * 25)
            print(f"    {cap:<25s}: {score:.3f}  {bar}")
        print("="*65 + "\n")