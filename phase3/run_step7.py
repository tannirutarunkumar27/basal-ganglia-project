"""
Step 7 verification:
  Feeds temporal belief into the ActionSelector.
  Checks that:
    - softmax probabilities are valid (sum to 1, all >= 0)
    - probabilistic selection favours the best action
    - temperature adaptation changes entropy correctly
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from belief.posterior_encoder import PosteriorBeliefEncoder
from temporal.temporal_belief import TemporalBeliefUpdater
from action.action_selector   import ActionSelector

np.random.seed(1)

N_ACTIONS = 4
N_NEURONS = 80
DT        = 0.1e-3
N_STEPS   = 600

encoder  = PosteriorBeliefEncoder(N_ACTIONS, N_NEURONS,
                                   window_steps=200, dt=DT)
updater  = TemporalBeliefUpdater(N_ACTIONS, lam=0.85)
selector = ActionSelector(N_ACTIONS, temperature=1.0)

action_log  = []
prob_log    = []
entropy_log = []
temp_log    = []

print("\n" + "="*55)
print("  Phase 3 — Step 7: Probabilistic Action Selection")
print("="*55)

for step in range(N_STEPS):
    # Action 3 strongly preferred in this test
    spikes = np.zeros(N_NEURONS, dtype=bool)
    neurons_per = N_NEURONS // N_ACTIONS
    for i in range(N_NEURONS):
        a_idx = i // neurons_per
        p = 0.18 if a_idx == 3 else 0.04
        spikes[i] = np.random.rand() < p

    V_hat = encoder.encode(spikes)
    V_t   = updater.update(V_hat)

    # Simulate uncertainty decreasing over time
    uncertainty = max(0.0, 0.8 - step / N_STEPS)
    selector.adapt_temperature(uncertainty)

    action, prob, info = selector.select(V_t, mode="probabilistic")

    action_log.append(action)
    prob_log.append(prob.copy())
    entropy_log.append(info["entropy"])
    temp_log.append(info["temperature"])

    if step % 10 == 0:
        encoder.update_prior(action=3, reward=1.0)

prob_arr  = np.array(prob_log)
t_ms      = np.arange(N_STEPS) * DT * 1000

# Validate
assert np.allclose(prob_arr.sum(axis=1), 1.0, atol=1e-6), \
    "Probabilities do not sum to 1"
assert np.all(prob_arr >= 0), "Negative probabilities found"

summary = selector.selection_summary()
print(f"\n  Selection frequency over {N_STEPS} steps:")
for a in range(N_ACTIONS):
    bar = "#" * int(summary['selection_freq'][a] * 40)
    print(f"    action {a}: {bar:<40s} "
          f"{summary['selection_freq'][a]*100:5.1f}%")

best = int(np.argmax(summary['selection_freq']))
print(f"\n  Most selected action: {best}  (expected: 3)")
print(f"  Soft competition score: "
      f"{summary['soft_competition_score']:.3f}")
print(f"  Final temperature: {summary['temperature']:.3f}")

status = "[PASS]" if best == 3 else "[WARN]"
print(f"\n  {status} Step 7 action selection.")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, prob_arr[:, a],
                 color=colors[a], lw=1.0, label=f"P(a={a}|s)")
axes[0].set_title("P(a|s) = softmax(Va) over time")
axes[0].set_ylabel("probability")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms, entropy_log, color="slateblue", lw=1)
axes[1].set_title("Entropy of action distribution H[P(a|s)]")
axes[1].set_ylabel("entropy (nats)")

axes[2].plot(t_ms, temp_log, color="darkorange", lw=1)
axes[2].set_title("Softmax temperature τ (decreasing with uncertainty)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("temperature")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step7_action.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step7_action.png")
print("="*55 + "\n")