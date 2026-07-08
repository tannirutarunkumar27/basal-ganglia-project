"""
Phase 6 full integration test — Phases 2-6.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2","phase3","phase4","phase5"]:
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
from rl_engine.rl_engine                     import RLEngine

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2
STATE_DIM = 2 * N_ACTIONS + 4   # matches RLEngine.build_state

print("\n" + "="*65)
print("  Phase 6 Integration: Full Pipeline Phases 2-6")
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
    n_actions=N_ACTIONS, theta_0=0.5, beta=0.4,
    kappa=0.3, refractory_ms=150.0,
    log_dir=os.path.join(HERE, "results"), dt=DT)
rl       = RLEngine(
    n_actions=N_ACTIONS, state_dim=STATE_DIM,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
    dt=DT)

logs = {k: [] for k in [
    "action","reward","delta_total","delta_prime",
    "D_pred","rho","pi_correct","entropy","da_response_num"]}
prev_action = None
prev_reward = None
cum_reward  = 0.0

print(f"  Running {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t = step * DT
    ctx_amp = 0.5e-9 * (np.sin(2*np.pi*5*t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    p3 = pipeline.step(spks["bayesian_layer"],
                        reward=prev_reward,
                        prev_action=prev_action)

    da = float(np.clip(
        1.0 + (net.pops["snc"].population_rate(50)-4.0)/20.0,
        0.1, 3.0))

    p4 = p4_ctrl.step(
        cortex_spikes=spks["cortex"],
        d1_spikes=spks["d1_msn"],
        d2_spikes=spks["d2_msn"],
        belief_scores=p3["V_combined"],
        U=p3["U"], C=p3["C"],
        dopamine_level=da)

    p5 = p5_ctrl.step(
        direct_inh=p4["direct_inh"],
        indirect_exc=p4["indirect_exc"],
        stn_global=p4["stn_global"],
        w_go=p4["w_go"], w_nogo=p4["w_nogo"], w_stn=p4["w_stn"],
        U=p3["U"], C=p3["C"],
        action_probs=p3["prob"],
        belief_scores=p3["V_combined"],
        conflict_score=p4["conflict_score"],
        t_ms=t*1000)

    action = (p5["released_action"]
              if p5["action_released"] else p3["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cum_reward += reward

    # Phase 6: RL update
    rl_out = rl.step(
        d1_spikes     = spks["d1_msn"],
        belief_scores = p3["V_combined"],
        raw_reward    = reward,
        action        = action,
        U             = p3["U"],
        C             = p3["C"],
        conflict_score= p4["conflict_score"],
        stn_burst     = p4["stn_burst"],
        dopamine_level= da,
        done          = False)

    # Feed delta_prime back into Phase 2 as dopamine signal
    net.step(cortex_input=ctx_in,
             dopamine_signal=rl_out["delta_prime"])

    # Update Phase 3 prior with delta
    pipeline.inject_dopamine_signal(rl_out["delta_prime"])

    prev_action = action
    prev_reward = reward

    logs["action"].append(action)
    logs["reward"].append(reward)
    logs["delta_total"].append(rl_out["delta_total"])
    logs["delta_prime"].append(rl_out["delta_prime"])
    logs["D_pred"].append(rl_out["D_pred"])
    logs["rho"].append(rl_out["rho"])
    logs["pi_correct"].append(float(rl_out["pi"][CORRECT]))
    logs["entropy"].append(rl_out["actor_entropy"])
    dr = {"burst":1,"flat":0,"dip":-1}
    logs["da_response_num"].append(
        dr.get(rl_out["da_response"], 0))

acc = np.mean(np.array(logs["action"]) == CORRECT) * 100
print(f"\n  Accuracy: {acc:.1f}%  |  Cumulative reward: {cum_reward:.1f}")

summary = rl.rl_summary()
print(f"\n  RL Engine summary:")
print(f"    Actor entropy         : {summary['actor']['entropy']:.4f}")
print(f"    Actor update count    : {summary['actor']['update_count']}")
print(f"    delta_total mean      : {summary['multi_critic']['delta_total_mean']:+.4f}")
print(f"    Predictive DA omega_1 : {summary['dopamine']['omega_1']:.3f}")
print(f"    Predictive DA omega_2 : {summary['dopamine']['omega_2']:.3f}")
print(f"    Risk rho              : {summary['risk']['rho']:.3f}")
print(f"    Best Q_risk action    : {summary['risk']['best_q_action']}")

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
t_ms_arr = np.arange(N_STEPS) * DT * 1000
smooth   = lambda v, w=100: np.convolve(v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms_arr, smooth(logs["pi_correct"]),
             color="forestgreen", lw=1.2, label=f"pi(a={CORRECT})")
axes[0].set_title("Actor policy pi(correct action)")
axes[0].set_ylabel("probability")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms_arr, smooth(logs["delta_total"]),
             color="steelblue", lw=1, label="delta_total")
axes[1].plot(t_ms_arr, smooth(logs["delta_prime"]),
             color="darkorange", lw=1, label="delta_prime")
axes[1].set_title("TD errors: delta_total (multi-critic) and delta_prime (predictive DA)")
axes[1].set_ylabel("delta")
axes[1].legend(fontsize=8)
axes[1].axhline(0, color="gray", ls="--", lw=0.5)

axes[2].plot(t_ms_arr, smooth(logs["D_pred"]),
             color="purple", lw=1)
axes[2].set_title("Predicted dopamine D_pred")
axes[2].set_ylabel("D_pred")
axes[2].axhline(0, color="gray", ls="--", lw=0.5)

axes[3].plot(t_ms_arr, smooth(logs["rho"]),
             color="coral", lw=1)
axes[3].set_title("Dynamic risk aversion rho (Step 18)")
axes[3].set_ylabel("rho")

axes[4].plot(t_ms_arr, np.cumsum(logs["reward"]),
             color="forestgreen", lw=1)
axes[4].set_title(f"Cumulative reward  (accuracy={acc:.1f}%)")
axes[4].set_xlabel("Time (ms)")
axes[4].set_ylabel("reward")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase6_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/phase6_integration.png")
print("="*65 + "\n")