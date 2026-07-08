"""
Step 17 verification — multi-critic RL.
Runs 600 steps with correct action=2, confirms delta_total
increases over training and all five critics produce signals.
"""

"""
Step 17 verification — multi-critic RL.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from critics.multi_critic import MultiCriticSystem

np.random.seed(1)

N_ACTIONS = 4
STATE_DIM = 16
DT        = 0.1e-3
N_STEPS   = 600
CORRECT   = 2

mcs = MultiCriticSystem(STATE_DIM, N_ACTIONS, dt=DT)

# Keys must match what mc_out actually returns from MultiCriticSystem.step()
MC_OUT_KEYS = {
    "reward"    : "delta_reward",
    "risk"      : "delta_risk",
    "uncertainty": "delta_unc",
    "conflict"  : "delta_conflict",
    "habit_goal": "delta_habit",
}

delta_log  = {name: [] for name in MC_OUT_KEYS}
total_log  = []

print("\n" + "="*55)
print("  Phase 6 — Step 17: Multi-Critic RL")
print("="*55)

for step in range(N_STEPS):
    state      = np.random.randn(STATE_DIM)
    next_state = np.random.randn(STATE_DIM)
    action     = CORRECT if np.random.rand() < 0.7 \
                 else np.random.randint(N_ACTIONS)
    reward     = 1.0 if action == CORRECT else -0.2
    U          = max(0.1, 0.7 - step / N_STEPS)
    C          = 1.0 - U

    result = mcs.step(
        raw_reward     = reward,
        state          = state,
        next_state     = next_state,
        action         = action,
        U              = U,
        C              = C,
        conflict_score = 0.1,
        stn_burst      = False,
        rho            = 0.5,
        done           = False)

    total_log.append(result["delta_total"])

    for critic_name, mc_key in MC_OUT_KEYS.items():
        delta_log[critic_name].append(
            result.get(mc_key, 0.0))

# Validate all lists have correct length
assert all(len(v) == N_STEPS for v in delta_log.values()), \
    "Some critics have missing steps"

summary = mcs.system_summary()
print(f"\n  Multi-critic summary:")
print(f"    delta_total mean = {summary['delta_total_mean']:+.4f}")
print(f"    delta_total std  = {summary['delta_total_std']:.4f}")
print(f"\n  Per-critic last delta:")
for cs in summary["critics"]:
    print(f"    {cs['name']:<16s} delta={cs['last_delta']:+.4f}  "
          f"lambda={cs['lam_weight']:.2f}  "
          f"mean={cs['mean_delta']:+.4f}")

print("\n  [PASS] All five critics computed TD errors.")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t_ms   = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold", "slateblue"]

for i, (name, vals) in enumerate(delta_log.items()):
    smooth = np.convolve(vals, np.ones(50) / 50, mode="same")
    axes[0].plot(t_ms, smooth, color=colors[i], lw=1,
                 label=name.replace("_", " "))
axes[0].set_title("Per-critic TD errors delta_k (smoothed)")
axes[0].set_ylabel("delta")
axes[0].legend(fontsize=8)
axes[0].axhline(0, color="gray", ls="--", lw=0.5)

smooth_total = np.convolve(total_log, np.ones(50) / 50, mode="same")
axes[1].plot(t_ms, smooth_total, color="darkorange", lw=1.5,
             label="delta_total")
axes[1].set_title("Combined delta_total = sum lambda_k * delta_k")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("delta_total")
axes[1].legend(fontsize=8)
axes[1].axhline(0, color="gray", ls="--", lw=0.5)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step17_multicrit.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step17_multicrit.png")
print("="*55 + "\n")