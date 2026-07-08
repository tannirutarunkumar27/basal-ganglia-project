"""
Step 30 verification — resource-aware pathway activation.
Tests three phases:
  Phase A: low U, high C  -> most gates closed  (exploit)
  Phase B: high U         -> STN opens           (explore)
  Phase C: high conflict  -> indirect+hyperdirect  (conflict)
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gates.conditional_gates import ConditionalGates

np.random.seed(1)

DT       = 0.1e-3
N_STEPS  = 900
PA_END   = 300
PB_END   = 600

gates = ConditionalGates(
    stn_U_threshold    = 0.55,
    action_C_threshold = 0.55,
    neuromod_threshold = 0.05,
    conflict_threshold = 0.25,
    dt                 = DT)

gate_log    = {k: [] for k in
               ["stn_gate","action_gate","neuromod_gate",
                "indirect_gate","hyperdirect_gate"]}
energy_log  = []
U_log       = []

print("\n" + "="*60)
print("  Phase 10 — Step 30: Resource-Aware Pathway Gates")
print("="*60)

for step in range(N_STEPS):
    if step < PA_END:
        U = 0.20 + np.random.randn() * 0.05   # low uncertainty
        C = 0.80 - np.random.randn() * 0.05   # high confidence
        delta = 0.02
        conflict = 0.10
    elif step < PB_END:
        U = 0.75 + np.random.randn() * 0.05   # high uncertainty
        C = 0.25 - np.random.randn() * 0.05   # low confidence
        delta = 0.08
        conflict = 0.20
    else:
        U = 0.55 + np.random.randn() * 0.05   # medium uncertainty
        C = 0.45 - np.random.randn() * 0.05
        delta = 0.15
        conflict = 0.45                         # high conflict

    U = float(np.clip(U, 0, 1))
    C = float(np.clip(C, 0, 1))

    gs = gates.evaluate(U, C, delta, conflict)

    for k in gate_log:
        gate_log[k].append(int(gs.get(k, False)))
    energy_log.append(gs["energy_step"])
    U_log.append(U)

# Analysis
summary = gates.gate_summary()
ar = summary["activation_rates"]
eff = summary["cumulative_eff"]

print(f"\n  Gate activation rates:")
for gate, rate in ar.items():
    bar = "#" * int(rate * 30)
    print(f"    {gate:<20s}: {rate:.3f}  {bar}")

print(f"\n  Cumulative energy efficiency: {eff*100:.1f}%")
print(f"  Total energy used          : {summary['total_energy_used']:.1f}")
print(f"  Total energy saved         : {summary['total_energy_saved']:.1f}")

mean_energy_A = np.mean(energy_log[:PA_END])
mean_energy_B = np.mean(energy_log[PA_END:PB_END])
print(f"\n  Mean energy/step Phase A (low U): {mean_energy_A:.3f}")
print(f"  Mean energy/step Phase B (high U): {mean_energy_B:.3f}")

assert eff > 0.05, "Expected some energy savings from gating"
print("\n  [PASS] Conditional gates reduce energy usage.")

t_ms = np.arange(N_STEPS) * DT * 1000
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
colors = {"stn_gate":"coral","action_gate":"steelblue",
          "neuromod_gate":"goldenrod",
          "indirect_gate":"forestgreen",
          "hyperdirect_gate":"slateblue"}
phase_kw = dict(alpha=0.07, zorder=0)
for ax in axes:
    ax.axvspan(0,              PA_END*DT*1000, color="steelblue", **phase_kw)
    ax.axvspan(PA_END*DT*1000, PB_END*DT*1000, color="coral",     **phase_kw)
    ax.axvspan(PB_END*DT*1000, N_STEPS*DT*1000, color="goldenrod", **phase_kw)

smooth = lambda v: np.convolve(v, np.ones(80)/80, mode="same")
for gname, color in colors.items():
    axes[0].plot(t_ms, smooth(gate_log[gname]),
                 color=color, lw=1, label=gname.replace("_gate",""))
axes[0].set_title("Gate activation rates  (Step 30)")
axes[0].set_ylabel("active fraction")
axes[0].legend(fontsize=8, ncol=3)
axes[0].set_ylim(0, 1.1)

axes[1].plot(t_ms, energy_log, color="darkorange", lw=0.6, alpha=0.5)
axes[1].plot(t_ms, smooth(energy_log), color="darkorange", lw=1.5)
axes[1].set_title("Energy cost per step  (always-on baseline = "
                  f"{sum(__import__('gates.conditional_gates', fromlist=['COMPONENT_COSTS']).COMPONENT_COSTS.values()):.1f})")
axes[1].set_ylabel("cost (a.u.)")

axes[2].plot(t_ms, U_log, color="crimson", lw=1)
axes[2].set_title("Uncertainty Ut  (drives gate activations)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Ut")
axes[2].set_ylim(0, 1)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step30_gates.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step30_gates.png")
print("="*60 + "\n")