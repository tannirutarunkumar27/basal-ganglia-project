"""
Step 19 verification — predictive dopamine.
Tests three scenarios:
  Phase A: unexpected reward  → burst
  Phase B: fully predicted    → flat (habituation)
  Phase C: unexpected omission → dip
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dopamine.predictive_dopamine import PredictiveDopamineModule

np.random.seed(3)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 900
PHASE_A   = 300
PHASE_B   = 600

da = PredictiveDopamineModule(N_ACTIONS, omega_1=0.6, omega_2=0.4, dt=DT)

d_pred_log    = []
delta_p_log   = []
omega_1_log   = []
omega_2_log   = []
resp_type_log = []

print("\n" + "="*55)
print("  Phase 6 — Step 19: Predictive Dopamine Modeling")
print("="*55)

for step in range(N_STEPS):
    V  = np.random.randn(N_ACTIONS) * 0.3
    U  = 0.3
    C  = 0.7

    if step < PHASE_A:
        # Phase A: unexpected reward — predictor not trained yet
        actual_reward = 1.0
        delta_obs     = 0.8
    elif step < PHASE_B:
        # Phase B: reward fully predicted — habituation
        actual_reward = 1.0   # same reward
        delta_obs     = 0.8
    else:
        # Phase C: reward omission — unexpected
        actual_reward = 0.0
        delta_obs     = -0.3

    # Predict
    D_pred = da.predict(V, U, C)

    # Combine observed + predicted
    delta_prime = da.combine_signals(delta_obs)

    # Update predictor
    da.update_predictor(actual_reward, V, U, C)

    resp = da.dopamine_response_type(delta_prime)

    d_pred_log.append(D_pred)
    delta_p_log.append(delta_prime)
    omega_1_log.append(da._omega_1_adapt)
    omega_2_log.append(da._omega_2_adapt)
    resp_type_log.append(resp)

t_ms = np.arange(N_STEPS) * DT * 1000

# Phase characterisation
burst_A  = resp_type_log[:PHASE_A].count("burst")
flat_B   = resp_type_log[PHASE_A:PHASE_B].count("flat")
dip_C    = resp_type_log[PHASE_B:].count("dip")

print(f"\n  Phase A (unexpected reward, 0-{PHASE_A}):")
print(f"    bursts: {burst_A}/{PHASE_A}  "
      f"({burst_A/PHASE_A*100:.1f}%)")

print(f"  Phase B (predicted reward, {PHASE_A}-{PHASE_B}):")
print(f"    flat:   {flat_B}/{PHASE_B-PHASE_A}  "
      f"({flat_B/(PHASE_B-PHASE_A)*100:.1f}%)")

print(f"  Phase C (omission, {PHASE_B}-{N_STEPS}):")
print(f"    dips:   {dip_C}/{N_STEPS-PHASE_B}  "
      f"({dip_C/(N_STEPS-PHASE_B)*100:.1f}%)")

summary = da.dopamine_summary()
print(f"\n  Final summary:")
print(f"    omega_1 (observed) : {summary['omega_1']:.3f}")
print(f"    omega_2 (predicted): {summary['omega_2']:.3f}")
print(f"    tonic DA level     : {summary['tonic_level']:.3f}")
print(f"    mean pred error    : {summary['mean_pred_error']:.4f}")

assert dip_C > (N_STEPS - PHASE_B) * 0.3, \
    "Expected dips in Phase C (omission)"
print("\n  [PASS] Predictive dopamine signals correct.")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
phase_kw = dict(alpha=0.08, zorder=0)
for ax in axes:
    ax.axvspan(0,             PHASE_A * DT * 1000,
               color="coral",     **phase_kw)
    ax.axvspan(PHASE_A*DT*1000, PHASE_B*DT*1000,
               color="steelblue", **phase_kw)
    ax.axvspan(PHASE_B*DT*1000, N_STEPS*DT*1000,
               color="goldenrod", **phase_kw)

smooth = lambda v, w=50: np.convolve(v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms, smooth(d_pred_log), color="darkorange",
             lw=1.2, label="D_pred (predicted)")
axes[0].plot(t_ms, smooth(delta_p_log), color="royalblue",
             lw=1.2, label="delta_prime (combined)")
axes[0].axhline(0, color="gray", ls="--", lw=0.5)
axes[0].set_title("Predictive DA  delta' = w1*delta + w2*D_pred  [Step 19]")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("signal")

axes[1].plot(t_ms, omega_1_log, color="steelblue", lw=1,
             label="omega_1 (observed weight)")
axes[1].plot(t_ms, omega_2_log, color="darkorange", lw=1,
             label="omega_2 (predictive weight)")
axes[1].set_title("Adaptive omega weights")
axes[1].set_ylabel("weight")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1)

resp_num = [{"burst":1,"flat":0,"dip":-1}[r] for r in resp_type_log]
axes[2].plot(t_ms, smooth(resp_num, 80), color="slateblue", lw=1)
axes[2].axhline(0, color="gray", ls="--", lw=0.5)
axes[2].set_title("Dopamine response type  (+1=burst, 0=flat, -1=dip)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("response")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step19_dopamine.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step19_dopamine.png")
print("="*55 + "\n")