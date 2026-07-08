"""
Step 13 verification.
Constructs a 4-action GPi gate, drives action 2 strongly
through the direct pathway, and confirms gate opens on a2.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gating.gpi_gate_engine import GPiGateEngine

np.random.seed(0)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 600
THRESHOLD = 0.6

gate = GPiGateEngine(N_ACTIONS, gpi_base=1.0, dt=DT)

gpi_log    = []
margin_log = []

print("\n" + "="*55)
print("  Phase 5 — Step 13: GPi Gate Engine")
print("="*55)

for step in range(N_STEPS):
    # Simulate: action 2 gets strong Go drive, others weak
    direct_inh   = np.array([0.05, 0.06, 0.55, 0.04])  # D1 per action
    indirect_exc = np.array([0.10, 0.12, 0.08, 0.11])  # indirect per action
    stn_global   = 0.05

    # Pathway weights: medium uncertainty (balanced)
    w_go, w_nogo, w_stn = 0.9, 0.5, 0.3

    gpi = gate.compute(direct_inh, indirect_exc,
                       stn_global, w_go, w_nogo, w_stn)
    gpi_log.append(gpi.copy())
    margin_log.append(gate.gate_margins(THRESHOLD).copy())

gpi_arr    = np.array(gpi_log)
margin_arr = np.array(margin_log)

summary = gate.gate_summary(THRESHOLD)
winner  = summary["winning_action"]

print(f"\n  Final GPi activity per action:")
for a in range(N_ACTIONS):
    status = " ← GATE OPEN" if (gpi_arr[-1, a] < THRESHOLD) else ""
    print(f"    action {a}: GPi={gpi_arr[-1,a]:.4f}  "
          f"margin={margin_arr[-1,a]:+.4f}{status}")

print(f"\n  Winning action: {winner}  (expected: 2)")
print(f"  Dominant pathway: {summary['dominant_pathway']}")
assert winner == 2, f"Expected action 2, got {winner}"
print("  [PASS] GPi gate selects correct action.")

# Pathway contribution breakdown
pc = summary["pathway_contribs"]
print(f"\n  Pathway contributions (action 2):")
print(f"    base  = {pc['base'][2]:.4f}")
print(f"    go    = -{pc['go'][2]:.4f}")
print(f"    nogo  = +{pc['nogo'][2]:.4f}")
print(f"    stn   = +{pc['stn'][2]:.4f}")
print(f"    net   = {pc['net'][2]:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t_ms   = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, gpi_arr[:, a],
                 color=colors[a], lw=1.2, label=f"GPi a{a}")
axes[0].axhline(THRESHOLD, color="red", ls="--", lw=1,
                label=f"threshold θ={THRESHOLD}")
axes[0].set_title("GPi activity per action channel (Step 13)")
axes[0].set_ylabel("GPi (a.u.)")
axes[0].legend(fontsize=8)

for a in range(N_ACTIONS):
    axes[1].plot(t_ms, margin_arr[:, a],
                 color=colors[a], lw=1, label=f"margin a{a}")
axes[1].axhline(0, color="red", ls="--", lw=0.8)
axes[1].set_title("Gate margin (θ - GPi_a) — positive = gate open")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("margin")
axes[1].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step13_gpi_gate.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step13_gpi_gate.png")
print("="*55 + "\n")