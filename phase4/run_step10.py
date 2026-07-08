"""
Step 10 verification — indirect pathway.
Tests that low dopamine increases No-Go (GPi excitation)
and suppresses the competing action correctly.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathways.indirect_pathway import IndirectPathway

np.random.seed(1)

N_ACTIONS = 4
N_D2_PER  = 20
DT        = 0.1e-3
N_STEPS   = 500

ip = IndirectPathway(N_ACTIONS, N_D2_PER, dt=DT)

print("\n" + "="*55)
print("  Phase 4 — Step 10: Indirect Pathway")
print("="*55)

results = {}
for da_label, da_level in [("low_DA", 0.3), ("high_DA", 2.0)]:
    ip.reset()
    gpi_log = []
    for step in range(N_STEPS):
        d2_spk = np.zeros(N_ACTIONS * N_D2_PER, dtype=bool)
        for i in range(N_ACTIONS * N_D2_PER):
            a = i // N_D2_PER
            p = 0.12 if a != 2 else 0.04   # competing actions fire
            d2_spk[i] = np.random.rand() < p
        gpi = ip.step(d2_spk, dopamine_level=da_level)
        gpi_log.append(gpi.copy())
    results[da_label] = np.array(gpi_log)

mean_low  = results["low_DA"][-100:, :].mean()
mean_high = results["high_DA"][-100:, :].mean()
print(f"\n  Mean GPi excitation (low  DA=0.3): {mean_low:.4f}")
print(f"  Mean GPi excitation (high DA=2.0): {mean_high:.4f}")
assert mean_low > mean_high, "Low DA should increase No-Go signal"
print("  [PASS] Indirect pathway No-Go modulation correct.")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t_ms = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]
for a in range(N_ACTIONS):
    axes[0].plot(t_ms, results["low_DA"][:, a],
                 color=colors[a], lw=1, label=f"a{a}")
    axes[1].plot(t_ms, results["high_DA"][:, a],
                 color=colors[a], lw=1, label=f"a{a}")
axes[0].set_title("GPi excitation (indirect) — low DA=0.3")
axes[1].set_title("GPi excitation (indirect) — high DA=2.0")
for ax in axes:
    ax.set_ylabel("excitation (a.u.)")
    ax.legend(fontsize=8)
axes[1].set_xlabel("Time (ms)")
plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step10_indirect.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step10_indirect.png")
print("="*55 + "\n")