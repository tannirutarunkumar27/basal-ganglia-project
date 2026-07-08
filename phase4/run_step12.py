"""
Step 12 verification — dynamic pathway weights.
Sweeps uncertainty from 0 to 1 and confirms:
  - wGo decreases with Ut
  - wNoGo increases with Ut
  - wSTN peaks at intermediate Ut (~0.5)
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weighting.dynamic_pathway_weights import DynamicPathwayWeights

print("\n" + "="*55)
print("  Phase 4 — Step 12: Dynamic Pathway Weights")
print("="*55)

dpw = DynamicPathwayWeights()

U_sweep = np.linspace(0.0, 1.0, 200)
w_go_log   = []
w_nogo_log = []
w_stn_log  = []
regime_log = []

for U in U_sweep:
    result = dpw.update(U)
    w_go_log.append(result["target_go"])      # use targets for clean sweep
    w_nogo_log.append(result["target_nogo"])
    w_stn_log.append(result["target_stn"])
    regime_log.append(result["regime"])

w_go_arr   = np.array(w_go_log)
w_nogo_arr = np.array(w_nogo_log)
w_stn_arr  = np.array(w_stn_log)

stn_peak_U = U_sweep[np.argmax(w_stn_arr)]
print(f"\n  wGo    at U=0.0: {w_go_arr[0]:.3f}  |  at U=1.0: {w_go_arr[-1]:.3f}")
print(f"  wNoGo  at U=0.0: {w_nogo_arr[0]:.3f}  |  at U=1.0: {w_nogo_arr[-1]:.3f}")
print(f"  wSTN   peak at U={stn_peak_U:.2f}")

assert w_go_arr[-1]   < w_go_arr[0],   "wGo should decrease with uncertainty"
assert w_nogo_arr[-1] > w_nogo_arr[0], "wNoGo should increase with uncertainty"
assert 0.3 < stn_peak_U < 0.7,         "wSTN should peak near U=0.5"

summary = dpw.weight_summary()
print(f"\n  Regime distribution:")
print(f"    Exploit  (Go dominant):  {summary['exploit_fraction']*100:.1f}%")
print(f"    Balanced (competition):  {summary['balanced_fraction']*100:.1f}%")
print(f"    Explore  (STN dominant): {summary['explore_fraction']*100:.1f}%")
print("\n  [PASS] Dynamic pathway weights correct.")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(U_sweep, w_go_arr,   color="steelblue",  lw=2, label="wGo")
axes[0].plot(U_sweep, w_nogo_arr, color="coral",      lw=2, label="wNoGo")
axes[0].plot(U_sweep, w_stn_arr,  color="darkorange", lw=2, label="wSTN")
axes[0].axvline(0.35, color="gray", ls=":", lw=1, label="regime boundaries")
axes[0].axvline(0.65, color="gray", ls=":", lw=1)
axes[0].set_xlabel("Uncertainty Ut")
axes[0].set_ylabel("pathway weight")
axes[0].set_title("Dynamic pathway weights as function of Ut")
axes[0].legend(fontsize=9)

colors_map = {"exploit": "steelblue", "balanced": "gold", "explore": "coral"}
for i, (U, r) in enumerate(zip(U_sweep, regime_log)):
    axes[1].axvspan(U, U + 1/200,
                    alpha=0.5, color=colors_map[r])
axes[1].set_xlabel("Uncertainty Ut")
axes[1].set_title("Regime: exploit (blue) | balanced (gold) | explore (coral)")
axes[1].set_yticks([])

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step12_weights.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step12_weights.png")
print("="*55 + "\n")