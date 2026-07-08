"""
Step 9 verification — direct pathway.
Simulates D1 firing with and without dopamine boost.
Checks that GPi inhibition increases with dopamine.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathways.direct_pathway import DirectPathway

np.random.seed(0)

N_ACTIONS = 4
N_D1_PER  = 20
DT        = 0.1e-3
N_STEPS   = 500

dp = DirectPathway(N_ACTIONS, N_D1_PER, dt=DT)

gpi_no_da  = []
gpi_with_da = []

print("\n" + "="*55)
print("  Phase 4 — Step 9: Direct Pathway")
print("="*55)

for trial, da_level in enumerate([0.5, 2.0]):
    dp.reset()
    gpi_log = []
    for step in range(N_STEPS):
        d1_spk = np.zeros(N_ACTIONS * N_D1_PER, dtype=bool)
        # Action 2 preferred
        for i in range(N_ACTIONS * N_D1_PER):
            a = i // N_D1_PER
            p = 0.15 if a == 2 else 0.04
            d1_spk[i] = np.random.rand() < p
        ctx_spk = np.random.rand(100) < 0.05
        gpi = dp.step(ctx_spk, d1_spk, dopamine_level=da_level)
        gpi_log.append(gpi.copy())
    if trial == 0:
        gpi_no_da   = np.array(gpi_log)
    else:
        gpi_with_da = np.array(gpi_log)

mean_no  = gpi_no_da[-100:,  2].mean()
mean_yes = gpi_with_da[-100:, 2].mean()

print(f"\n  Action 2 GPi inhibition (no DA,  DA=0.5): {mean_no:.4f}")
print(f"  Action 2 GPi inhibition (with DA, DA=2.0): {mean_yes:.4f}")
assert mean_yes > mean_no, "Dopamine should increase GPi inhibition"
print("  [PASS] Dopamine boosts direct pathway Go signal.")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t_ms = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]
for a in range(N_ACTIONS):
    axes[0].plot(t_ms, gpi_no_da[:, a],
                 color=colors[a], lw=1, label=f"a{a}")
    axes[1].plot(t_ms, gpi_with_da[:, a],
                 color=colors[a], lw=1, label=f"a{a}")
axes[0].set_title("GPi inhibition — low dopamine (DA=0.5)")
axes[1].set_title("GPi inhibition — high dopamine (DA=2.0)")
for ax in axes:
    ax.set_ylabel("inhibition (a.u.)")
    ax.legend(fontsize=8)
axes[1].set_xlabel("Time (ms)")
plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step9_direct.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step9_direct.png")
print("="*55 + "\n")