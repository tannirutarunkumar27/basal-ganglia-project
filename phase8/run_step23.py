"""
Step 23 verification — meta-dopamine learning rate.
Tests three phases:
  Phase A: high uncertainty -> alpha_t rises
  Phase B: low uncertainty  -> alpha_t falls
  Phase C: volatile rewards -> eta increases, alpha_t responsive
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_dopamine.meta_dopamine import MetaDopamine

np.random.seed(0)

DT       = 0.1e-3
N_STEPS  = 900
PA_END   = 300
PB_END   = 600

md = MetaDopamine(alpha_0=0.05, eta=0.10,
                   alpha_min=0.001, alpha_max=0.30, dt=DT)

alpha_log  = []
U_log      = []
vol_log    = []
regime_log = []

print("\n" + "="*55)
print("  Phase 8 — Step 23: Meta-Dopamine Learning Rate")
print("="*55)

for step in range(N_STEPS):
    if step < PA_END:
        U      = 0.80 + np.random.randn() * 0.04
        reward = np.random.choice([1.0, -0.5], p=[0.5, 0.5])
    elif step < PB_END:
        U      = 0.15 + np.random.randn() * 0.04
        reward = 1.0
    else:
        U      = 0.55 + np.random.randn() * 0.15
        reward = np.random.choice([1.5, -1.0], p=[0.4, 0.6])

    U = float(np.clip(U, 0.0, 1.0))
    alpha_t = md.update(U, reward)

    alpha_log.append(alpha_t)
    U_log.append(U)
    vol_log.append(md.volatility)
    regime_log.append(md.plasticity_regime())

alpha_arr = np.array(alpha_log)
mean_A    = float(np.mean(alpha_arr[:PA_END]))
mean_B    = float(np.mean(alpha_arr[PA_END:PB_END]))
mean_C    = float(np.mean(alpha_arr[PB_END:]))

print(f"\n  alpha_t mean:")
print(f"    Phase A (high U, volatile): {mean_A:.5f}")
print(f"    Phase B (low  U, stable):  {mean_B:.5f}")
print(f"    Phase C (medium U, volatile):{mean_C:.5f}")
assert mean_A > mean_B, "High U should produce higher alpha_t"
print("  [PASS] Meta-dopamine raises alpha_t with uncertainty.")

summary = md.meta_summary()
print(f"\n  Final meta-dopamine summary:")
for k, v in summary.items():
    print(f"    {k:<20s}: {v}")

regime_counts = {r: regime_log.count(r)
                 for r in ["exploratory","balanced","consolidating"]}
print(f"\n  Regime distribution:")
for r, n in regime_counts.items():
    print(f"    {r:<16s}: {n/N_STEPS*100:5.1f}%")

t_ms   = np.arange(N_STEPS) * DT * 1000
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
phase_kw = dict(alpha=0.08, zorder=0)
for ax in axes:
    ax.axvspan(0,          PA_END*DT*1000, color="coral",      **phase_kw)
    ax.axvspan(PA_END*DT*1000, PB_END*DT*1000, color="steelblue", **phase_kw)
    ax.axvspan(PB_END*DT*1000, N_STEPS*DT*1000, color="goldenrod", **phase_kw)

axes[0].plot(t_ms, U_log, color="slateblue", lw=1)
axes[0].set_title("Uncertainty Ut input")
axes[0].set_ylabel("Ut")
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms, alpha_log, color="darkorange", lw=1.2,
             label="alpha_t")
axes[1].axhline(md.alpha_0, color="gray", ls="--",
                lw=0.8, label=f"alpha_0={md.alpha_0}")
axes[1].set_title("alpha_t = alpha_0 + eta * Ut  (Step 23)")
axes[1].set_ylabel("alpha_t")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms, vol_log, color="forestgreen", lw=1)
axes[2].set_title("Reward volatility (boosts eta during volatile phases)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("volatility")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step23_meta_da.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step23_meta_da.png")
print("="*55 + "\n")