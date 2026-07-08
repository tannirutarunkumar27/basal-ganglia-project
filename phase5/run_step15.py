"""
Step 15 verification — thalamocortical relay + explainability logging.
Simulates 600 ms, releases actions, generates text explanations,
saves the decision log, and prints the full summary.
"""

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gating.gpi_gate_engine            import GPiGateEngine
from threshold.adaptive_threshold      import AdaptiveThreshold
from relay.thalamocortical_relay        import ThalamocorticalRelay
from xai_logging.explainability_logger import ExplainabilityLogger

np.random.seed(5)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 600
CORRECT   = 2

gate   = GPiGateEngine(N_ACTIONS, gpi_base=1.0, dt=DT)
thresh = AdaptiveThreshold(N_ACTIONS, theta_0=0.5,
                            beta=0.4, kappa=0.3, dt=DT)
relay  = ThalamocorticalRelay(N_ACTIONS, refractory_ms=100.0, dt=DT)
logger = ExplainabilityLogger(
    N_ACTIONS, log_dir=os.path.join(HERE, "results"))

action_log     = []
release_log    = []
confidence_log = []
theta_log      = []
motor_log      = []

print("\n" + "="*60)
print("  Phase 5 — Step 15: Thalamocortical Relay + Logging")
print("="*60)

for step in range(N_STEPS):
    t_ms     = step * DT * 1000
    progress = min(step / N_STEPS, 1.0)

    # ── Normalised signals [0, 1] ─────────────────────────────────
    # Action 2 Go inhibition grows from 0.1 -> 0.85 (learning)
    d1_a2 = 0.10 + 0.75 * progress

    # direct_inh: strong Go on action 2, weak on others
    direct_inh   = np.array([0.08, 0.09, d1_a2, 0.07])

    # indirect_exc: competing actions have moderate No-Go
    indirect_exc = np.array([0.15, 0.12, 0.06, 0.14])

    # stn_global: decreases as certainty grows
    stn_global   = 0.10 * (1.0 - 0.7 * progress)

    # Uncertainty decreases over time (learning)
    U = max(0.10, 0.75 - progress * 0.65)
    C = 1.0 - U

    # Action probabilities from Bayesian pipeline (simulated)
    probs  = np.array([0.08, 0.08, 0.76, 0.08]) * C \
             + np.ones(4) * 0.25 * U
    probs /= probs.sum()

    w_go, w_nogo, w_stn = 0.9, 0.4, 0.25

    # Step 13: GPi gate (normalised inputs)
    gpi = gate.compute(direct_inh, indirect_exc,
                       stn_global, w_go, w_nogo, w_stn)
    pc  = gate.pathway_contributions()

    # Step 14: adaptive threshold
    t_out = thresh.update(U, C)
    theta = t_out["theta"]

    # Step 15: relay
    record = relay.step(
        gpi_activity     = gpi,
        threshold        = theta,
        action_probs     = probs,
        U                = U,
        C                = C,
        conflict_score   = 0.05 + 0.25 * (1.0 - progress),
        pathway_contribs = pc,
        t_ms             = t_ms)

    logger.log(record)

    action_log.append(record["released_action"])
    release_log.append(record["action_released"])
    confidence_log.append(record["release_confidence"])
    theta_log.append(theta)
    motor_out = record.get("motor_output")
    motor_log.append(motor_out.copy()
                     if isinstance(motor_out, np.ndarray)
                     else np.zeros(N_ACTIONS))

# Sample explanation
releases = logger.get_releases_only(10)
if releases:
    names = ["reach_left", "reach_right", "press_button", "wait"]
    print("\n  Sample decision explanation (last release):")
    print(logger.generate_text_explanation(releases[-1], names))
else:
    print("\n  [WARN] Still no releases — check gpi_base vs threshold.")

# Pathway attribution
attr = logger.pathway_attribution_report(200)
if attr:
    print("\n  Pathway attribution (last 200 steps):")
    for pathway, vals in attr.items():
        print(f"    {pathway:<22s}: "
              f"{[f'{v:.3f}' for v in vals]}")

log_path = logger.save_log("step15_decisions.json")
print(f"\n  Decision log saved: {log_path}")
logger.print_summary()

released_actions = [a for a in action_log if a is not None]
acc = (sum(a == CORRECT for a in released_actions)
       / max(len(released_actions), 1))
print(f"\n  Release accuracy (action {CORRECT}): {acc*100:.1f}%")

# ── Plots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
t_ms_arr  = np.arange(N_STEPS) * DT * 1000
motor_arr = np.array(motor_log)
colors    = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms_arr, motor_arr[:, a],
                 color=colors[a], lw=1.0, label=f"a{a}")
axes[0].set_title("Motor cortex output per action channel (Step 15)")
axes[0].set_ylabel("motor signal")
axes[0].legend(fontsize=8)

axes[1].plot(t_ms_arr, theta_log, color="darkorange",
             lw=1.2, label="theta_t (adaptive)")
axes[1].set_title("Adaptive threshold  theta_t = theta_0 + b*Ut - k*Ct")
axes[1].set_ylabel("threshold")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms_arr, confidence_log, color="royalblue", lw=1)
axes[2].set_title("Release confidence score")
axes[2].set_ylabel("confidence")
axes[2].set_ylim(0, 1)

released_t = [t_ms_arr[i] for i in range(N_STEPS)
              if release_log[i] and action_log[i] is not None]
released_a = [action_log[i] for i in range(N_STEPS)
              if release_log[i] and action_log[i] is not None]

if released_t:
    axes[3].scatter(released_t, released_a,
                    c=[colors[a % 4] for a in released_a],
                    s=15, alpha=0.8, zorder=3)
axes[3].axhline(CORRECT, color="red", ls="--",
                lw=0.8, label=f"correct a{CORRECT}")
axes[3].set_title(f"Released actions  (accuracy={acc*100:.1f}%)")
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("action")
axes[3].set_yticks(range(N_ACTIONS))
axes[3].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step15_relay.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step15_relay.png")
print("=" * 60 + "\n")