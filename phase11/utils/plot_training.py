"""
plot_training — produces publication-quality training plots
for all key metrics across episodes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_training_curves(all_ep_metrics: list,
                          step_log      : list,
                          results_dir   : str = "results",
                          dt            : float = 0.1e-3) -> str:

    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle("Phase 11 — Full Online Training Loop Metrics",
                 fontsize=13, y=0.98)

    n_ep    = len(all_ep_metrics)
    ep_nums = list(range(1, n_ep + 1))

    # ── Episode-level metrics ──────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ep_nums,
            [m["accuracy"] * 100 for m in all_ep_metrics],
            color="forestgreen", lw=1.5, marker="o", ms=5)
    ax.set_title("Accuracy per episode (%)")
    ax.set_ylabel("accuracy %")
    ax.set_xlabel("episode")
    ax.set_ylim(0, 100)
    ax.axhline(75, color="gray", ls="--", lw=0.8)

    ax = axes[0, 1]
    ax.plot(ep_nums,
            [m["mean_reward"] for m in all_ep_metrics],
            color="steelblue", lw=1.5, marker="o", ms=5)
    ax.set_title("Mean reward per episode")
    ax.set_ylabel("mean reward")
    ax.set_xlabel("episode")
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    ax = axes[1, 0]
    ax.plot(ep_nums,
            [m["total_energy_nJ"] for m in all_ep_metrics],
            color="darkorange", lw=1.5, marker="s", ms=5)
    ax.set_title("Total energy per episode (nJ)")
    ax.set_ylabel("energy nJ")
    ax.set_xlabel("episode")

    ax = axes[1, 1]
    ax.plot(ep_nums,
            [m["efficiency_score"] for m in all_ep_metrics],
            color="purple", lw=1.5, marker="^", ms=5)
    ax.set_title("Efficiency score (accuracy / norm_energy)")
    ax.set_ylabel("efficiency")
    ax.set_xlabel("episode")

    # ── Step-level metrics (last episode) ─────────────────────
    smooth = lambda v, w=200: np.convolve(
        v, np.ones(w)/w, mode="same")

    if step_log:
        n_steps  = len(step_log)
        t_ms_arr = np.arange(n_steps) * dt * 1000

        rewards = [s["reward"]      for s in step_log]
        deltas  = [s["delta_total"] for s in step_log]
        alphas  = [s["alpha_t"]     for s in step_log]
        Us      = [s["U"]           for s in step_log]
        spikes  = [s["n_spikes_opt"]for s in step_log]
        energies= [s["energy_pJ"]   for s in step_log]

        ax = axes[2, 0]
        ax.plot(t_ms_arr, smooth(rewards), color="forestgreen", lw=1)
        ax.set_title("Reward (smoothed) — last episode")
        ax.set_ylabel("reward")
        ax.axhline(0, color="gray", ls="--", lw=0.5)

        ax = axes[2, 1]
        ax.plot(t_ms_arr, smooth(deltas), color="steelblue", lw=1,
                label="delta_total")
        ax.set_title("Multi-critic TD error delta_total — last episode")
        ax.set_ylabel("delta")
        ax.axhline(0, color="gray", ls="--", lw=0.5)

        ax = axes[3, 0]
        ax.plot(t_ms_arr, smooth(alphas), color="darkorange", lw=1)
        ax.set_title("Meta-DA learning rate alpha_t — last episode")
        ax.set_ylabel("alpha_t")

        ax = axes[3, 1]
        ax.plot(t_ms_arr, smooth(Us), color="crimson", lw=1,
                label="Ut")
        ax.set_title("Uncertainty Ut — last episode")
        ax.set_ylabel("Ut")
        ax.set_ylim(0, 1)

        ax = axes[4, 0]
        ax.plot(t_ms_arr, smooth(spikes), color="slateblue", lw=1)
        ax.set_title("Optimized spike count — last episode")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("spikes/step")

        ax = axes[4, 1]
        ax.plot(t_ms_arr, smooth(energies, 100),
                color="coral", lw=1)
        ax.set_title("Energy per step pJ — last episode")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("energy pJ")

    plt.tight_layout()
    path = os.path.join(results_dir, "phase11_training_curves.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path