"""
Step 14 verification — confidence-based adaptive threshold.
Tests three regimes:
  Phase A (0–200 ms) : high U → cautious  (θ rises)
  Phase B (200–500 ms): low U → decisive  (θ falls)
  Phase C (500–700 ms): reversal → θ spikes temporarily
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from threshold.adaptive_threshold import AdaptiveThreshold

np.random.seed(3)

N_ACTIONS   = 4
DT          = 0.1e-3
N_STEPS     = 700

# Phase boundary indices (in steps, NOT milliseconds)
PHASE_A_END = 200     # steps 0-199   : high U
PHASE_B_END = 500     # steps 200-499 : low  U
# steps 500-699 : reversal

thresh = AdaptiveThreshold(N_ACTIONS, theta_0=0.5,
                            beta=0.4, kappa=0.3, dt=DT)

theta_log  = []
U_log      = []
regime_log = []
sat_log    = []

print("\n" + "="*55)
print("  Phase 5 — Step 14: Adaptive Threshold")
print("="*55)

for step in range(N_STEPS):
    if step < PHASE_A_END:
        U = 0.75 + np.random.randn() * 0.05    # high uncertainty
    elif step < PHASE_B_END:
        U = 0.20 + np.random.randn() * 0.05    # low uncertainty
    else:
        U = 0.65 + abs(np.random.randn()) * 0.1 # reversal

    U = float(np.clip(U, 0.0, 1.0))
    C = 1.0 - U

    result = thresh.update(U, C)
    theta_log.append(result["theta"])
    U_log.append(U)
    regime_log.append(result["regime"])
    sat_log.append(thresh.speed_accuracy_tradeoff())

theta_arr = np.array(theta_log)
U_arr     = np.array(U_log)
t_ms      = np.arange(N_STEPS) * DT * 1000

# Use correct step-based slices
mean_cautious = float(np.mean(theta_arr[:PHASE_A_END]))
mean_decisive = float(np.mean(theta_arr[PHASE_A_END:PHASE_B_END]))

print(f"\n  Phase A (steps 0–{PHASE_A_END},  high U): "
      f"mean theta = {mean_cautious:.4f}")
print(f"  Phase B (steps {PHASE_A_END}–{PHASE_B_END}, low  U): "
      f"mean theta = {mean_decisive:.4f}")

assert not np.isnan(mean_cautious), "mean_cautious is NaN — check slice"
assert not np.isnan(mean_decisive), "mean_decisive is NaN — check slice"
assert mean_cautious > mean_decisive, (
    f"High uncertainty should raise threshold: "
    f"{mean_cautious:.4f} vs {mean_decisive:.4f}")
print("  [PASS] Threshold rises with uncertainty.")

summary = thresh.threshold_summary()
print(f"\n  Final threshold summary:")
print(f"    mean theta  = {summary['mean_theta']:.4f}")
print(f"    std  theta  = {summary['std_theta']:.4f}")
print(f"    SAT score   = {summary['sat_score']:+.4f}  "
      f"(+1=accuracy, -1=speed)")
print(f"    adaptive b  = {summary['beta']:.4f}")
print(f"    adaptive k  = {summary['kappa']:.4f}")

regime_counts = {r: regime_log.count(r)
                 for r in ["cautious", "balanced", "decisive"]}
for r, n in regime_counts.items():
    print(f"    regime {r:<10s}: {n / N_STEPS * 100:5.1f}%")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
phase_kw = dict(zorder=0, alpha=0.10)
for ax in axes:
    ax.axvspan(t_ms[0],              t_ms[PHASE_A_END - 1],
               color="coral",        **phase_kw)
    ax.axvspan(t_ms[PHASE_A_END],    t_ms[PHASE_B_END - 1],
               color="steelblue",    **phase_kw)
    ax.axvspan(t_ms[PHASE_B_END],    t_ms[-1],
               color="goldenrod",    **phase_kw)

axes[0].plot(t_ms, U_arr,   color="crimson",   lw=1, label="Ut")
axes[0].plot(t_ms, 1-U_arr, color="royalblue", lw=1, label="Ct")
axes[0].set_title("Input: uncertainty Ut and confidence Ct")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("value")

axes[1].plot(t_ms, theta_arr, color="darkorange", lw=1.2, label="theta_t")
axes[1].axhline(0.5, color="gray", ls="--", lw=0.8, label="theta_0=0.5")
axes[1].set_title("Adaptive threshold  theta_t = theta_0 + b*Ut - k*Ct")
axes[1].set_ylabel("threshold")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms, sat_log, color="slateblue", lw=1)
axes[2].axhline(0, color="gray", ls="--", lw=0.8)
axes[2].set_title("Speed-accuracy tradeoff  (+ = accuracy, - = speed)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("SAT score")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step14_threshold.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step14_threshold.png")
print("=" * 55 + "\n")