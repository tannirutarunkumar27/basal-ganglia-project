"""
Full Phase 3 integration test:
  Connects the BayesianReasoningPipeline to the Phase 2
  BGNetwork and runs 1 second of joint simulation.
"""
import sys, os
HERE     = os.path.dirname(os.path.abspath(__file__))
PHASE2   = os.path.join(os.path.dirname(HERE), "phase2")
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE2)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.bg_network import BGNetwork
from integration.bayesian_reasoning_pipeline import BayesianReasoningPipeline

np.random.seed(42)

DT         = 0.1e-3
SIM_TIME   = 1.0        # 1 second
N_STEPS    = int(SIM_TIME / DT)
N_ACTIONS  = 4

print("\n" + "="*60)
print("  Phase 3 Integration: BayesianPipeline + BGNetwork")
print("="*60)

# Build Phase 2 network
net = BGNetwork(dt=DT)

# Build Phase 3 pipeline
# Use bayesian_layer population (60 neurons, 4 actions = 15/action)
pipeline = BayesianReasoningPipeline(
    n_actions       = N_ACTIONS,
    n_neurons_total = net.pops["bayesian_layer"].N,
    window_steps    = 100,
    lam             = 0.85,
    dt              = DT,
)

# Logging
U_log      = []
C_log      = []
action_log = []
reward_log = []
V_log      = []

prev_action = None
prev_reward = None

print(f"  Simulating {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t = step * DT

    # Build cortex input (task-modulated 5 Hz drive)
    ctx_amp = 0.5e-9 * (np.sin(2 * np.pi * 5 * t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    # Run BG network one step
    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    # Feed bayesian_layer spikes into reasoning pipeline
    out = pipeline.step(
        spike_vector  = spks["bayesian_layer"],
        selection_mode= "probabilistic",
        reward        = prev_reward,
        prev_action   = prev_action,
    )

    # Simulate task reward (action 2 is correct in this demo)
    correct_action = 2
    reward = 1.0 if out["action"] == correct_action else -0.1
    prev_action = out["action"]
    prev_reward = reward

    U_log.append(out["U"])
    C_log.append(out["C"])
    action_log.append(out["action"])
    reward_log.append(reward)
    V_log.append(out["V_combined"].copy())

# Summary
V_arr     = np.array(V_log)
t_ms      = np.arange(N_STEPS) * DT * 1000
acc       = float(np.mean(np.array(action_log) == correct_action))

print(f"\n  Final pipeline summary:")
s = pipeline.pipeline_summary()
print(f"    Last action        : {s['last_action']}")
print(f"    Last U             : {s['last_U']:.3f}")
print(f"    Last C             : {s['last_C']:.3f}")
print(f"    Total selections   : {s['total_selections']}")
print(f"    Accuracy (action 2): {acc*100:.1f}%")
print(f"    Learned prior      : "
      f"{[f'{p:.3f}' for p in s['prior']]}")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, V_arr[:, a],
                 color=colors[a], lw=0.6, alpha=0.8, label=f"a{a}")
axes[0].set_title("Combined belief Va (temporal + memory)")
axes[0].set_ylabel("belief")
axes[0].legend(fontsize=7)

axes[1].plot(t_ms, U_log, color="crimson",   lw=1, label="Ut")
axes[1].plot(t_ms, C_log, color="royalblue", lw=1, label="Ct")
axes[1].set_title("Uncertainty Ut and confidence Ct")
axes[1].set_ylabel("value [0,1]")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1)

axes[2].scatter(t_ms[::10], action_log[::10],
                s=4, c=[colors[a] for a in action_log[::10]],
                alpha=0.7)
axes[2].axhline(correct_action, color="red", ls="--", lw=0.8,
                label=f"correct action ({correct_action})")
axes[2].set_title(f"Selected actions  (accuracy={acc*100:.1f}%)")
axes[2].set_ylabel("action")
axes[2].legend(fontsize=8)
axes[2].set_yticks(range(N_ACTIONS))

cum_reward = np.cumsum(reward_log)
axes[3].plot(t_ms, cum_reward, color="forestgreen", lw=1)
axes[3].set_title("Cumulative reward")
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("reward")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase3_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Integration plot saved: results/phase3_integration.png")
print("="*60 + "\n")