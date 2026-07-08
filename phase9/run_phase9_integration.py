"""
Phase 9 full integration test — Phases 2-9.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2","phase3","phase4","phase5",
          "phase6","phase7","phase8"]:
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
from plasticity.plasticity_manager           import PlasticityManager
from controllers.neuromodulator_controller   import NeuromodulatorController
from integration.reasoning_pipeline          import ReasoningPipeline

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2
STATE_DIM = 2 * N_ACTIONS + 4
NAMES     = ["reach_left","reach_right","press_button","wait"]

print("\n" + "="*65)
print("  Phase 9 Integration: Full Pipeline Phases 2-9")
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
rl       = RLEngine(n_actions=N_ACTIONS, state_dim=STATE_DIM,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS, dt=DT)
pop_sizes = {k: net.pops[k].N for k in
             ["cortex","d1_msn","d2_msn","gpi","gpe"]}
plast    = PlasticityManager(pop_sizes=pop_sizes,
    n_actions=N_ACTIONS, base_alpha=0.05, dt=DT)
nm_ctrl  = NeuromodulatorController(
    alpha_0=0.05, eta=0.10, dt=DT)

# Phase 9
reasoning = ReasoningPipeline(
    n_actions=N_ACTIONS, action_names=NAMES,
    log_dir=os.path.join(HERE, "results"), dt=DT)

logs = {k: [] for k in ["action","reward","expl_conf",
                          "n_rules","attn_dom","acc_running"]}
prev_action   = None
prev_reward   = None
cum_reward    = 0.0
correct_count = 0
last_explanation = ""

EXPLAIN_EVERY = 1000   # generate full text every 100ms

print(f"  Running {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t    = step * DT
    t_ms = t * 1000

    ctx_amp = 0.5e-9 * (np.sin(2*np.pi*5*t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)
    p3 = pipeline.step(spks["bayesian_layer"],
                        reward=prev_reward, prev_action=prev_action)

    da       = float(np.clip(
        1.0 + (net.pops["snc"].population_rate(50)-4.0)/20.0,
        0.1, 3.0))
    snc_rate = net.pops["snc"].population_rate(50)

    p4 = p4_ctrl.step(
        cortex_spikes=spks["cortex"], d1_spikes=spks["d1_msn"],
        d2_spikes=spks["d2_msn"], belief_scores=p3["V_combined"],
        U=p3["U"], C=p3["C"], dopamine_level=da)

    p5 = p5_ctrl.step(
        direct_inh=p4["direct_inh"], indirect_exc=p4["indirect_exc"],
        stn_global=p4["stn_global"], w_go=p4["w_go"],
        w_nogo=p4["w_nogo"], w_stn=p4["w_stn"],
        U=p3["U"], C=p3["C"], action_probs=p3["prob"],
        belief_scores=p3["V_combined"],
        conflict_score=p4["conflict_score"], t_ms=t_ms)

    action = (p5["released_action"]
              if p5["action_released"] else p3["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cum_reward += reward
    if action == CORRECT:
        correct_count += 1

    rl_out = rl.step(
        d1_spikes=spks["d1_msn"], belief_scores=p3["V_combined"],
        raw_reward=reward, action=action, U=p3["U"], C=p3["C"],
        conflict_score=p4["conflict_score"], stn_burst=p4["stn_burst"],
        dopamine_level=da, done=False)

    nm_out = nm_ctrl.step(
        delta_prime=rl_out["delta_prime"], U=p3["U"], C=p3["C"],
        reward=reward, rho=rl_out["rho"],
        conflict_score=p4["conflict_score"], snc_rate_hz=snc_rate)

    plast.step(spks, rl_out["delta_prime"]*nm_out["Mt"],
               nm_out["alpha_t"])

    # Phase 9: reasoning and explainability
    gm = p5.get("gate_margins", np.zeros(N_ACTIONS))
    if not isinstance(gm, np.ndarray):
        gm = np.zeros(N_ACTIONS)

    do_explain = (step % EXPLAIN_EVERY == 0)

    p9 = reasoning.step(
        V_combined     = p3["V_combined"],
        U              = p3["U"],
        C              = p3["C"],
        Q_risk         = rl_out.get("Q_risk", np.zeros(N_ACTIONS)),
        conflict_score = p4["conflict_score"],
        stn_burst      = p4["stn_burst"],
        direct_inh     = p4["direct_inh"],
        indirect_exc   = p4["indirect_exc"],
        stn_global     = p4["stn_global"],
        gate_margins   = gm,
        DA             = nm_out["DA"],
        ht5            = nm_out["5HT"],
        NE             = nm_out["NE"],
        alpha_t        = nm_out["alpha_t"],
        Mt             = nm_out["Mt"],
        rho            = rl_out["rho"],
        reward         = reward,
        gate_action    = action,
        t_ms           = t_ms,
        explain        = do_explain)

    if do_explain and p9["full_explanation"]:
        last_explanation = p9["full_explanation"]

    net.step(cortex_input=ctx_in,
             dopamine_signal=rl_out["delta_prime"] * nm_out["DA"])
    pipeline.inject_dopamine_signal(rl_out["delta_prime"]*nm_out["Mt"])
    p4_ctrl.apply_reward(reward, action)
    prev_action = action
    prev_reward = reward

    logs["action"].append(action)
    logs["reward"].append(reward)
    logs["expl_conf"].append(p9["explanation_conf"])
    logs["n_rules"].append(p9["n_rules_fired"])
    logs["attn_dom"].append(p9.get("dominant_signal",""))
    logs["acc_running"].append(correct_count / (step + 1))

acc = correct_count / N_STEPS * 100
print(f"\n  Accuracy   : {acc:.1f}%")
print(f"  Cum reward : {cum_reward:.1f}")
print(f"\n  Last full explanation (t={N_STEPS//EXPLAIN_EVERY*EXPLAIN_EVERY/10:.0f}ms):")
print("  " + "-"*60)
if last_explanation:
    for line in last_explanation.split("\n"):
        print(f"  {line}")
print("  " + "-"*60)

log_path = reasoning.save_explanations("phase9_integration_log.json")
print(f"\n  Explanations saved: {log_path}")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
t_ms_arr = np.arange(N_STEPS) * DT * 1000
smooth   = lambda v, w=100: np.convolve(v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms_arr, logs["acc_running"],
             color="forestgreen", lw=1.2)
axes[0].set_title(f"Running accuracy  (final={acc:.1f}%)")
axes[0].set_ylabel("accuracy")
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms_arr, smooth(logs["expl_conf"]),
             color="darkorange", lw=1.2)
axes[1].set_title("Explanation confidence (Step 28)")
axes[1].set_ylabel("confidence")
axes[1].set_ylim(0, 1)

axes[2].plot(t_ms_arr, smooth(logs["n_rules"]),
             color="slateblue", lw=1)
axes[2].set_title("Number of symbolic rules fired per step (Step 25)")
axes[2].set_ylabel("rules")

axes[3].plot(t_ms_arr, np.cumsum(logs["reward"]),
             color="forestgreen", lw=1)
axes[3].set_title("Cumulative reward")
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("reward")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase9_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/phase9_integration.png")
print("="*65 + "\n")