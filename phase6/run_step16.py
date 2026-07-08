"""
Step 16 verification — striatal actor.
Simulates D1 activity with action 2 preferred and confirms
pi(a|s) converges to favour action 2 after reward updates.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actor.striatal_actor import StriatalActor

np.random.seed(0)

N_ACTIONS = 4
N_D1_PER  = 20
DT        = 0.1e-3
N_STEPS   = 800
CORRECT   = 2

actor = StriatalActor(N_ACTIONS, N_D1_PER, dt=DT)

pi_log   = []
pref_log = []

print("\n" + "="*55)
print("  Phase 6 — Step 16: Striatal Actor")
print("="*55)

for step in range(N_STEPS):
    # Simulate D1 spikes: action 2 preferred
    d1_spikes = np.zeros(N_ACTIONS * N_D1_PER, dtype=bool)
    for i in range(N_ACTIONS * N_D1_PER):
        a = i // N_D1_PER
        p = 0.15 if a == CORRECT else 0.04
        d1_spikes[i] = np.random.rand() < p

    d1_rate = actor.encode_d1_activity(d1_spikes, dopamine_level=1.5)
    pi      = actor.compute_policy(d1_rate=d1_rate)
    action  = actor.select_action(mode="sample")

    # Reward signal
    reward  = 1.0 if action == CORRECT else -0.1
    # Simulated delta (will come from multi-critic in Step 17)
    delta   = reward - 0.5   # simple RPE for test
    actor.update(delta, action)
    # Temperature adapts with simulated uncertainty
    actor.adapt_temperature(U=max(0.1, 0.7 - step / N_STEPS))

    pi_log.append(pi.copy())
    pref_log.append(actor.action_preference.copy())

pi_arr   = np.array(pi_log)
pref_arr = np.array(pref_log)

final_pi   = pi_arr[-50:].mean(axis=0)
best       = int(np.argmax(final_pi))
entropy    = actor.actor_summary()["entropy"]

print(f"\n  Final mean policy (last 50 steps):")
for a in range(N_ACTIONS):
    bar = "#" * int(final_pi[a] * 40)
    print(f"    pi(a={a}) = {final_pi[a]:.3f}  {bar}")
print(f"\n  Best action: {best}  (expected: {CORRECT})")
print(f"  Policy entropy: {entropy:.4f}")
assert best == CORRECT, f"Expected action {CORRECT}, got {best}"
print("  [PASS] Actor converges to correct action.")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t_ms   = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, pi_arr[:, a],
                 color=colors[a], lw=1, label=f"pi(a={a})")
axes[0].set_title("Policy pi(a|s) = softmax(D1 activity)  [Step 16]")
axes[0].set_ylabel("probability")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)

for a in range(N_ACTIONS):
    axes[1].plot(t_ms, pref_arr[:, a],
                 color=colors[a], lw=1, label=f"pref a{a}")
axes[1].set_title("Action preference (learned by actor)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("preference")
axes[1].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step16_actor.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step16_actor.png")
print("="*55 + "\n")