"""
Phase 4 full integration test.
Connects Phase 2 (BGNetwork) + Phase 3 (BayesianPipeline)
+ Phase 4 (BGPathwayController) for 1 second of simulation.
"""
import sys, os
HERE   = os.path.dirname(os.path.abspath(__file__))
P2     = os.path.join(os.path.dirname(HERE), "phase2")
P3     = os.path.join(os.path.dirname(HERE), "phase3")
sys.path.insert(0, HERE)
sys.path.insert(0, P2)
sys.path.insert(0, P3)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.bg_network                        import BGNetwork
from integration.bayesian_reasoning_pipeline  import BayesianReasoningPipeline
from integration.bg_pathway_controller        import BGPathwayController

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2     # rewarded action in this demo

print("\n" + "="*65)
print("  Phase 4 Integration: Pathways + Reasoning + BGNetwork")
print("="*65)

net      = BGNetwork(dt=DT)
pipeline = BayesianReasoningPipeline(
    n_actions=N_ACTIONS,
    n_neurons_total=net.pops["bayesian_layer"].N,
    window_steps=100, lam=0.85, dt=DT)
ctrl     = BGPathwayController(
    n_actions=N_ACTIONS,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
    n_d2_per_action=net.pops["d2_msn"].N // N_ACTIONS,
    conflict_eps=0.3, dt=DT)

# Logging
log = {k: [] for k in ["U","C","w_go","w_nogo","w_stn",
                        "gpi","regime","conflict","action","reward"]}

prev_action = None
prev_reward = None
cumulative_reward = 0.0

print(f"  Running {SIM_TIME*1000:.0f} ms simulation...")

for step in range(N_STEPS):
    t = step * DT
    ctx_amp = 0.5e-9 * (np.sin(2 * np.pi * 5 * t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    # Phase 2
    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    # Phase 3
    p3_out = pipeline.step(
        spike_vector   = spks["bayesian_layer"],
        selection_mode = "probabilistic",
        reward         = prev_reward,
        prev_action    = prev_action)

    U = p3_out["U"]
    C = p3_out["C"]

    # Estimate dopamine from SNc spike rate
    snc_rate = net.pops["snc"].population_rate(50)
    da_level = 1.0 + (snc_rate - 4.0) / 20.0
    da_level = float(np.clip(da_level, 0.1, 3.0))

    # Phase 4
    p4_out = ctrl.step(
        cortex_spikes  = spks["cortex"],
        d1_spikes      = spks["d1_msn"],
        d2_spikes      = spks["d2_msn"],
        belief_scores  = p3_out["V_combined"],
        U              = U,
        C              = C,
        dopamine_level = da_level)

    # Use Phase 4 gate action if available, else Phase 3
    action = (p4_out["released_action"]
              if p4_out["gate_open"] else p3_out["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cumulative_reward += reward
    ctrl.apply_reward(delta=reward, action=action)
    pipeline.inject_dopamine_signal(reward)

    prev_action = action
    prev_reward = reward

    log["U"].append(U)
    log["C"].append(C)
    log["w_go"].append(p4_out["w_go"])
    log["w_nogo"].append(p4_out["w_nogo"])
    log["w_stn"].append(p4_out["w_stn"])
    log["gpi"].append(p4_out["gpi_activity"].min())
    log["regime"].append(p4_out["regime"])
    log["conflict"].append(p4_out["conflict_score"])
    log["action"].append(action)
    log["reward"].append(reward)

acc = np.mean(np.array(log["action"]) == CORRECT) * 100
print(f"\n  Accuracy (action {CORRECT}):  {acc:.1f}%")
print(f"  Cumulative reward:          {cumulative_reward:.1f}")
regime_counts = {r: log["regime"].count(r) for r in
                 ["exploit","balanced","explore"]}
total_steps = len(log["regime"])
for r, n in regime_counts.items():
    print(f"  Regime {r:<10s}: {n/total_steps*100:5.1f}%")

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
t_ms = np.arange(N_STEPS) * DT * 1000

axes[0].plot(t_ms, log["U"], color="crimson",   lw=1, label="Ut")
axes[0].plot(t_ms, log["C"], color="royalblue", lw=1, label="Ct")
axes[0].set_title("Uncertainty and confidence")
axes[0].legend(fontsize=8); axes[0].set_ylim(0,1)

axes[1].plot(t_ms, log["w_go"],   color="steelblue",  lw=1, label="wGo")
axes[1].plot(t_ms, log["w_nogo"], color="coral",      lw=1, label="wNoGo")
axes[1].plot(t_ms, log["w_stn"],  color="darkorange", lw=1, label="wSTN")
axes[1].set_title("Dynamic pathway weights (Step 12)")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms, log["conflict"], color="slateblue", lw=1)
axes[2].axhline(y=0.3, color="red", ls="--", lw=0.8,
                label="conflict threshold")
axes[2].set_title("Conflict score (hyperdirect trigger)")
axes[2].legend(fontsize=8)

axes[3].plot(t_ms, log["gpi"], color="darkorange", lw=1)
axes[3].set_title("Min GPi activity (lower = action released)")
axes[3].set_ylabel("GPi")

axes[4].plot(t_ms, np.cumsum(log["reward"]), color="forestgreen", lw=1)
axes[4].set_title(f"Cumulative reward (accuracy={acc:.1f}%)")
axes[4].set_xlabel("Time (ms)")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase4_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Integration plot saved: results/phase4_integration.png")
print("="*65 + "\n")