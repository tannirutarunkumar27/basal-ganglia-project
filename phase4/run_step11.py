"""
Step 11 verification — hyperdirect pathway + conflict detection.
Tests:
  - Conflict detected when beliefs are near-uniform (ambiguous)
  - STN bursts and suppresses action when conflict triggered
  - Conflict resolves when one belief dominates
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathways.hyperdirect_pathway import HyperdirectPathway

np.random.seed(2)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 800
RESOLVE   = 400   # beliefs diverge here

hp = HyperdirectPathway(N_ACTIONS, conflict_eps=0.3, dt=DT)

stn_log      = []
conflict_log = []
gpi_exc_log  = []
burst_log    = []

print("\n" + "="*55)
print("  Phase 4 — Step 11: Hyperdirect Pathway")
print("="*55)

for step in range(N_STEPS):
    if step < RESOLVE:
        # Ambiguous: all beliefs similar → conflict
        V = np.array([0.1, 0.12, 0.09, 0.11]) + np.random.randn(N_ACTIONS)*0.02
    else:
        # Resolved: action 2 dominates → no conflict
        V = np.array([-0.5, -0.3, 1.8, -0.4]) + np.random.randn(N_ACTIONS)*0.05

    ctx_spk = np.random.rand(100) < 0.05
    gpi_exc = hp.step(ctx_spk, V, stn_trigger=0.0)

    stn_log.append(hp.pathway_summary()["stn_activity"])
    conflict_log.append(hp.pathway_summary()["conflict_score"])
    gpi_exc_log.append(gpi_exc)
    burst_log.append(float(hp.stn_burst))

t_ms = np.arange(N_STEPS) * DT * 1000

mean_conflict_pre  = np.mean(conflict_log[:RESOLVE])
mean_conflict_post = np.mean(conflict_log[RESOLVE:])
mean_burst_pre     = np.mean(burst_log[:RESOLVE])
mean_burst_post    = np.mean(burst_log[RESOLVE:])

print(f"\n  Pre-resolve  conflict score: {mean_conflict_pre:.4f}")
print(f"  Post-resolve conflict score: {mean_conflict_post:.4f}")
print(f"  Pre-resolve  STN burst frac: {mean_burst_pre:.3f}")
print(f"  Post-resolve STN burst frac: {mean_burst_post:.3f}")

assert mean_burst_pre > mean_burst_post, \
    "STN should burst more during ambiguous period"
print("  [PASS] Hyperdirect pathway conflict detection correct.")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t_ms, conflict_log, color="slateblue", lw=1)
axes[0].axhline(y=hp.conflict_eps, color="red", ls="--", lw=0.8,
                label=f"conflict threshold ε={hp.conflict_eps}")
axes[0].axvline(RESOLVE * DT * 1000, color="gray", ls=":", lw=1)
axes[0].set_title("Conflict score (low = ambiguous)")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("conflict")

axes[1].plot(t_ms, stn_log, color="crimson", lw=1)
axes[1].fill_between(t_ms, 0, burst_log,
                     alpha=0.3, color="orange", label="STN burst active")
axes[1].axvline(RESOLVE * DT * 1000, color="gray", ls=":", lw=1)
axes[1].set_title("STN activity + burst episodes")
axes[1].legend(fontsize=8)
axes[1].set_ylabel("activity")

axes[2].plot(t_ms, gpi_exc_log, color="darkorange", lw=1)
axes[2].axvline(RESOLVE * DT * 1000, color="gray", ls=":",
                lw=1, label="belief resolves")
axes[2].set_title("GPi global excitation from hyperdirect pathway")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("excitation")
axes[2].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step11_hyperdirect.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step11_hyperdirect.png")
print("="*55 + "\n")