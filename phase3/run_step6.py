"""
Step 6 verification:
  Tests temporal belief updating with a reversal scenario.
  At step 300, reward shifts from action 2 to action 0.
  The temporal updater should track the shift.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from belief.posterior_encoder  import PosteriorBeliefEncoder
from temporal.temporal_belief  import TemporalBeliefUpdater
from temporal.memory_trace     import MemoryTrace

np.random.seed(0)

N_ACTIONS = 4
N_NEURONS = 80
WINDOW    = 200
DT        = 0.1e-3
N_STEPS   = 800
REVERSAL  = 400     # action preference shifts here

encoder  = PosteriorBeliefEncoder(N_ACTIONS, N_NEURONS,
                                   window_steps=WINDOW, dt=DT)
updater  = TemporalBeliefUpdater(N_ACTIONS, lam=0.85)
memory   = MemoryTrace(N_ACTIONS, dt=DT)

V_hat_log      = []
V_temporal_log = []
memory_log     = []

print("\n" + "="*55)
print("  Phase 3 — Step 6: Temporal Belief Updating")
print("="*55)

for step in range(N_STEPS):
    # Before reversal: action 2 is preferred
    # After reversal:  action 0 is preferred
    preferred = 0 if step >= REVERSAL else 2

    spikes = np.zeros(N_NEURONS, dtype=bool)
    neurons_per_action = N_NEURONS // N_ACTIONS
    for i in range(N_NEURONS):
        a_idx = i // neurons_per_action
        p = 0.15 if a_idx == preferred else 0.04
        spikes[i] = np.random.rand() < p

    # Step 5: instantaneous belief
    V_hat = encoder.encode(spikes)

    # Step 6: temporal update
    V_t = updater.update(V_hat, adaptive_lam=True)

    # Memory trace update
    V_mem = memory.step(V_t)

    V_hat_log.append(V_hat.copy())
    V_temporal_log.append(V_t.copy())
    memory_log.append(V_mem.copy())

    # Update prior with reward
    if step % 10 == 0:
        encoder.update_prior(preferred, reward=1.0)

V_hat_arr = np.array(V_hat_log)
V_tmp_arr = np.array(V_temporal_log)
V_mem_arr = np.array(memory_log)

# Evaluate tracking
post_reversal = V_tmp_arr[REVERSAL + 100:, :]
best_post = int(np.argmax(post_reversal.mean(axis=0)))
print(f"\n  Pre-reversal  preferred action: 2")
print(f"  Post-reversal preferred action: 0")
print(f"  Tracked action after reversal:  {best_post}")
print(f"  Volatility: {updater._volatility:.4f}")
print(f"  Dominant memory timescale: {memory.dominant_timescale()}")

status = "[PASS]" if best_post == 0 else "[WARN]"
print(f"\n  {status} Temporal belief tracking.")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
t_ms = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, V_hat_arr[:, a],
                 color=colors[a], alpha=0.6, lw=0.8, label=f"a{a}")
axes[0].axvline(REVERSAL * DT * 1000, color="red", ls="--", lw=1)
axes[0].set_title("Step 5: instantaneous belief Va_hat(t)")
axes[0].set_ylabel("log-posterior")
axes[0].legend(fontsize=8)

for a in range(N_ACTIONS):
    axes[1].plot(t_ms, V_tmp_arr[:, a],
                 color=colors[a], lw=1.2, label=f"a{a}")
axes[1].axvline(REVERSAL * DT * 1000, color="red", ls="--", lw=1)
axes[1].set_title("Step 6: temporal belief Va(t) = λVa(t-1) + (1-λ)Va_hat")
axes[1].set_ylabel("belief score")
axes[1].legend(fontsize=8)

for a in range(N_ACTIONS):
    axes[2].plot(t_ms, V_mem_arr[:, a],
                 color=colors[a], lw=1.2, alpha=0.9, label=f"a{a}")
axes[2].axvline(REVERSAL * DT * 1000, color="red", ls="--", lw=1,
                label="reversal point")
axes[2].set_title("Memory trace output (3 timescales combined)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("memory signal")
axes[2].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step6_temporal.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step6_temporal.png")
print("="*55 + "\n")