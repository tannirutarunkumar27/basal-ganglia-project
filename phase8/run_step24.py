"""
Step 24 verification — multi-neuromodulator fusion.
Tests three environmental phases with different dominant modulators:
  Phase A: high reward, low uncertainty -> DA dominant
  Phase B: high risk/punishment         -> 5-HT dominant
  Phase C: high surprise/arousal        -> NE dominant
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuromodulators.dopamine_module       import DopamineModule
from neuromodulators.serotonin_module      import SerotoninModule
from neuromodulators.norepinephrine_module import NorepinephrineModule
from fusion.neuromodulator_fusion          import NeuromodulatorFusion

np.random.seed(1)

DT       = 0.1e-3
N_STEPS  = 900
PA_END   = 300
PB_END   = 600

da_mod  = DopamineModule(dt=DT)
ht5_mod = SerotoninModule(dt=DT)
ne_mod  = NorepinephrineModule(dt=DT)
fusion  = NeuromodulatorFusion(omega_d=0.5, omega_s=0.3,
                                omega_n=0.2, dt=DT)

DA_log   = []
HT5_log  = []
NE_log   = []
Mt_log   = []
lr_log   = []
temp_log = []
rho_log  = []
wg_log   = []
wd_log   = []
omega_log = []

print("\n" + "="*55)
print("  Phase 8 — Step 24: Multi-Neuromodulator Fusion")
print("="*55)

for step in range(N_STEPS):
    if step < PA_END:
        # High reward environment: DA dominant
        reward   = 1.0 + np.random.randn() * 0.2
        delta    = 0.8
        U        = 0.2
        rho      = 0.2
        conflict = 0.1
        volatility = 0.05
    elif step < PB_END:
        # High risk/punishment: 5-HT dominant
        reward   = np.random.choice([1.0, -1.5], p=[0.4, 0.6])
        delta    = -0.3
        U        = 0.5
        rho      = 0.9
        conflict = 0.3
        volatility = 0.3
    else:
        # High surprise/novelty: NE dominant
        reward   = np.random.choice([2.0, -0.5], p=[0.3, 0.7])
        delta    = np.random.randn() * 1.5
        U        = 0.7
        rho      = 0.5
        conflict = 0.6
        volatility = 0.6

    # Update each neuromodulator
    DA  = da_mod.update(delta, snc_rate_hz=4.0 + 2.0*float(delta > 0))
    DA_norm = da_mod.normalised()
    ht5 = ht5_mod.update(reward, rho, conflict)
    NE  = ne_mod.update(delta, U, volatility)

    # Compute fused signal and control outputs
    ctrl = fusion.compute_control_signals(
        alpha_t_base = 0.05,
        U            = U,
        DA_t         = DA_norm,
        ht5_t        = ht5,
        NE_t         = NE)

    # Adapt fusion weights
    fusion.adapt_weights(reward, DA_norm, ht5, NE)

    DA_log.append(DA_norm)
    HT5_log.append(ht5)
    NE_log.append(NE)
    Mt_log.append(ctrl["Mt"])
    lr_log.append(ctrl["learning_rate"])
    temp_log.append(ctrl["explore_temp"])
    rho_log.append(ctrl["rho_adjustment"])
    wg_log.append(ctrl["w_go_scale"])
    wd_log.append(ctrl["w_nogo_scale"])
    omega_log.append((ctrl["omega_d"],
                      ctrl["omega_s"],
                      ctrl["omega_n"]))

omega_arr = np.array(omega_log)
t_ms      = np.arange(N_STEPS) * DT * 1000

mean_DA_A  = np.mean(DA_log[:PA_END])
mean_HT5_B = np.mean(HT5_log[PA_END:PB_END])
mean_NE_C  = np.mean(NE_log[PB_END:])

print(f"\n  Phase A (DA dominant):  mean DA  = {mean_DA_A:.3f}")
print(f"  Phase B (5HT dominant): mean 5HT = {mean_HT5_B:.3f}")
print(f"  Phase C (NE dominant):  mean NE  = {mean_NE_C:.3f}")

summary = fusion.fusion_summary()
print(f"\n  Final fusion summary:")
print(f"    Mt            = {summary['Mt']:.4f}")
print(f"    omega_d (DA)  = {summary['omega_d']:.3f}")
print(f"    omega_s (5HT) = {summary['omega_s']:.3f}")
print(f"    omega_n (NE)  = {summary['omega_n']:.3f}")
print(f"    dominant      = {summary['dominant']}")
print(f"\n  Last control signals:")
for k, v in summary["last_ctrl"].items():
    if isinstance(v, float):
        print(f"    {k:<20s}: {v:.4f}")

print("\n  [PASS] Multi-neuromodulator fusion computed.")

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
phase_kw = dict(alpha=0.08, zorder=0)
for ax in axes:
    ax.axvspan(0,              PA_END*DT*1000,  color="steelblue", **phase_kw)
    ax.axvspan(PA_END*DT*1000, PB_END*DT*1000,  color="coral",     **phase_kw)
    ax.axvspan(PB_END*DT*1000, N_STEPS*DT*1000, color="goldenrod",  **phase_kw)

smooth = lambda v, w=50: np.convolve(v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms, smooth(DA_log),  color="steelblue",  lw=1, label="DA")
axes[0].plot(t_ms, smooth(HT5_log), color="coral",      lw=1, label="5-HT")
axes[0].plot(t_ms, smooth(NE_log),  color="goldenrod",  lw=1, label="NE")
axes[0].plot(t_ms, smooth(Mt_log),  color="slateblue",  lw=1.5,
             ls="--", label="Mt (fused)")
axes[0].set_title("Neuromodulator levels and fused signal Mt (Step 24)")
axes[0].set_ylabel("level")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.1)

axes[1].plot(t_ms, omega_arr[:, 0], color="steelblue", lw=1, label="omega_d")
axes[1].plot(t_ms, omega_arr[:, 1], color="coral",     lw=1, label="omega_s")
axes[1].plot(t_ms, omega_arr[:, 2], color="goldenrod", lw=1, label="omega_n")
axes[1].set_title("Adaptive fusion weights (shift toward most predictive)")
axes[1].set_ylabel("omega")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1)

axes[2].plot(t_ms, smooth(lr_log),   color="purple",  lw=1, label="learning rate")
axes[2].plot(t_ms, smooth(temp_log), color="darkorange", lw=1, label="explore temp")
axes[2].set_title("Mt-regulated learning rate and exploration temperature")
axes[2].set_ylabel("value")
axes[2].legend(fontsize=8)

axes[3].plot(t_ms, smooth(wg_log), color="steelblue", lw=1, label="wGo scale")
axes[3].plot(t_ms, smooth(wd_log), color="coral",     lw=1, label="wNoGo scale")
axes[3].plot(t_ms, smooth(rho_log),color="forestgreen",lw=1, label="rho adj")
axes[3].set_title("Pathway balance (wGo/wNoGo) and risk sensitivity (rho) from Mt")
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("scale")
axes[3].legend(fontsize=8)
axes[3].axhline(0, color="gray", ls="--", lw=0.5)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step24_fusion.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step24_fusion.png")
print("="*55 + "\n")