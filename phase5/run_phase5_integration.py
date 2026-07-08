"""
Phase 5 full integration test.
Connects Phase 2 (BGNetwork) + Phase 3 (BayesianPipeline)
+ Phase 4 (BGPathwayController) + Phase 5 (ActionGatingController).
"""

"""
Phase 5 full integration test — Phases 2 through 5.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2", "phase3", "phase4"]:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), p))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.bg_network                      import BGNetwork
from integration.bayesian_reasoning_pipeline import BayesianReasoningPipeline
from integration.bg_pathway_controller       import BGPathwayController
from integration.action_gating_controller    import ActionGatingController

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2

print("\n" + "="*65)
print("  Phase 5 Integration: Full Pipeline Phases 2-5")
print("="*65)

net      = BGNetwork(dt=DT)
pipeline = BayesianReasoningPipeline(
    n_actions=N_ACTIONS,
    n_neurons_total=net.pops["bayesian_layer"].N,
    window_steps=100, lam=0.85, dt=DT)
p4_ctrl  = BGPathwayController(
    n_actions=N_ACTIONS,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
    n_d2_per_action=net.pops["d2_msn"].N // N_ACTIONS,
    conflict_eps=0.3, dt=DT)
p5_ctrl  = ActionGatingController(
    n_actions=N_ACTIONS,
    theta_0=0.5, beta=0.4, kappa=0.3,
    refractory_ms=150.0,
    log_dir=os.path.join(HERE, "results"),
    dt=DT)

logs = {k: [] for k in
        ["action", "reward", "conf", "theta",
         "U", "C", "margin", "motor"]}
prev_action = None
prev_reward = None
cum_reward  = 0.0

print(f"  Running {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t    = step * DT
    t_ms = t * 1000

    ctx_amp = 0.5e-9 * (np.sin(2 * np.pi * 5 * t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    # Phase 2
    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    # Phase 3
    p3 = pipeline.step(spks["bayesian_layer"],
                        reward=prev_reward,
                        prev_action=prev_action)

    # Dopamine level from SNc
    da = float(np.clip(
        1.0 + (net.pops["snc"].population_rate(50) - 4.0) / 20.0,
        0.1, 3.0))

    # Phase 4
    p4 = p4_ctrl.step(
        cortex_spikes = spks["cortex"],
        d1_spikes     = spks["d1_msn"],
        d2_spikes     = spks["d2_msn"],
        belief_scores = p3["V_combined"],
        U=p3["U"], C=p3["C"],
        dopamine_level=da)

    # Phase 5
    p5 = p5_ctrl.step(
        direct_inh    = p4["direct_inh"],
        indirect_exc  = p4["indirect_exc"],
        stn_global    = p4["stn_global"],
        w_go          = p4["w_go"],
        w_nogo        = p4["w_nogo"],
        w_stn         = p4["w_stn"],
        U             = p3["U"],
        C             = p3["C"],
        action_probs  = p3["prob"],
        belief_scores = p3["V_combined"],
        conflict_score= p4["conflict_score"],
        t_ms          = t_ms)

    # Final action: prefer Phase 5 gate, fallback to Phase 3
    action = (p5["released_action"]
              if p5["action_released"] else p3["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cum_reward += reward
    p4_ctrl.apply_reward(reward, action)
    pipeline.inject_dopamine_signal(reward)

    prev_action = action
    prev_reward = reward

    logs["action"].append(action)
    logs["reward"].append(reward)
    logs["conf"].append(p5["release_confidence"])
    logs["theta"].append(p5["threshold"])
    logs["U"].append(p5["U"])
    logs["C"].append(p5["C"])
    margins = p5.get("gate_margins", [0] * N_ACTIONS)
    logs["margin"].append(max(margins) if margins else 0.0)
    mo = p5.get("motor_output")
    logs["motor"].append(mo.copy()
                         if isinstance(mo, np.ndarray)
                         else np.zeros(N_ACTIONS))

# Explanation
print("\n  Final action explanation:")
print(p5_ctrl.get_explanation(
    ["reach_left", "reach_right", "press_button", "wait"]))

log_path = p5_ctrl.save_log("phase5_full_integration.json")
print(f"\n  Log saved: {log_path}")
p5_ctrl.logger.print_summary()

acc = np.mean(np.array(logs["action"]) == CORRECT) * 100
print(f"\n  Accuracy: {acc:.1f}%   Cumulative reward: {cum_reward:.1f}")

# Plot
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
t_ms_arr  = np.arange(N_STEPS) * DT * 1000
motor_arr = np.array(logs["motor"])
colors    = ["steelblue", "coral", "forestgreen", "gold"]

axes[0].plot(t_ms_arr, logs["U"], color="crimson",   lw=1, label="Ut")
axes[0].plot(t_ms_arr, logs["C"], color="royalblue", lw=1, label="Ct")
axes[0].set_title("Uncertainty and confidence (Phase 3 → Phase 5)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms_arr, logs["theta"], color="darkorange",
             lw=1.2, label="theta_t")
axes[1].set_title("Adaptive threshold theta_t (Step 14)")
axes[1].set_ylabel("threshold")
axes[1].legend(fontsize=8)

for a in range(N_ACTIONS):
    axes[2].plot(t_ms_arr, motor_arr[:, a],
                 color=colors[a], lw=0.8, label=f"a{a}")
axes[2].set_title("Motor cortex output (Step 15 relay)")
axes[2].set_ylabel("signal")
axes[2].legend(fontsize=8)

axes[3].scatter(t_ms_arr[::5], logs["action"][::5],
                c=[colors[a % 4] for a in logs["action"][::5]],
                s=4, alpha=0.7)
axes[3].axhline(CORRECT, color="red", ls="--",
                lw=0.8, label=f"correct a{CORRECT}")
axes[3].set_title(f"Selected actions  (accuracy={acc:.1f}%)")
axes[3].set_ylabel("action")
axes[3].set_yticks(range(N_ACTIONS))
axes[3].legend(fontsize=8)

axes[4].plot(t_ms_arr, np.cumsum(logs["reward"]),
             color="forestgreen", lw=1)
axes[4].set_title("Cumulative reward")
axes[4].set_xlabel("Time (ms)")
axes[4].set_ylabel("reward")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase5_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/phase5_integration.png")
print("=" * 65 + "\n")