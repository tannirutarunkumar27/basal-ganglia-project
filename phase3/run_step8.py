"""
Step 8 verification:
  Tests uncertainty and confidence computation across
  three phases:
    Phase A (0-200 ms)   : high noise, unstable beliefs -> high U
    Phase B (200-500 ms) : one action dominant          -> U decreases
    Phase C (500-700 ms) : sudden reversal              -> U spikes
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from belief.posterior_encoder    import PosteriorBeliefEncoder
from temporal.temporal_belief    import TemporalBeliefUpdater
from action.action_selector      import ActionSelector
from uncertainty.uncertainty_module import UncertaintyModule

np.random.seed(7)

N_ACTIONS = 4
N_NEURONS = 80
DT        = 0.1e-3
N_STEPS   = 700

encoder   = PosteriorBeliefEncoder(N_ACTIONS, N_NEURONS,
                                    window_steps=200, dt=DT)
updater   = TemporalBeliefUpdater(N_ACTIONS, lam=0.85)
selector  = ActionSelector(N_ACTIONS, temperature=1.0)
unc_mod   = UncertaintyModule(N_ACTIONS, variance_window=50)

U_log     = []
C_log     = []
alpha_log = []
gate_log  = []
temp_log  = []
risk_log  = []
action_log = []

print("\n" + "="*55)
print("  Phase 3 — Step 8: Uncertainty & Confidence")
print("="*55)

for step in range(N_STEPS):
    # Three phases of behaviour
    if step < 200:
        # Phase A: random noise, no dominant action
        probs = [0.06] * N_ACTIONS
    elif step < 500:
        # Phase B: action 1 becomes dominant
        probs = [0.04, 0.16, 0.04, 0.04]
    else:
        # Phase C: sudden reversal to action 3
        probs = [0.04, 0.04, 0.04, 0.16]

    spikes = np.zeros(N_NEURONS, dtype=bool)
    neurons_per = N_NEURONS // N_ACTIONS
    for i in range(N_NEURONS):
        a_idx = i // neurons_per
        spikes[i] = np.random.rand() < probs[a_idx]

    V_hat  = encoder.encode(spikes)
    V_t    = updater.update(V_hat)
    uc     = unc_mod.update(V_t)

    # Adapt selector temperature from uncertainty
    selector.adapt_temperature(uc["U"])
    action, prob, sel_info = selector.select(
        V_t, mode="probabilistic",
        temp_override=uc["exploration_temp"])

    U_log.append(uc["U"])
    C_log.append(uc["C"])
    alpha_log.append(uc["learning_rate_factor"])
    gate_log.append(uc["gate_threshold_offset"])
    temp_log.append(uc["exploration_temp"])
    risk_log.append(uc["risk_aversion"])
    action_log.append(action)

    if step % 20 == 0:
        preferred = int(np.argmax(probs))
        encoder.update_prior(preferred, reward=1.0)

U_arr     = np.array(U_log)
C_arr     = np.array(C_log)
t_ms      = np.arange(N_STEPS) * DT * 1000

print(f"\n  Phase A (0-200ms)  mean U = "
      f"{np.mean(U_arr[:2000]):.3f}  (expected: high ~0.5+)")
print(f"  Phase B (200-500ms) mean U = "
      f"{np.mean(U_arr[2000:5000]):.3f}  (expected: lower)")
print(f"  Phase C (500-700ms) mean U = "
      f"{np.mean(U_arr[5000:]):.3f}  (expected: high at reversal)")

phase_b_u = np.mean(U_arr[2000:5000])
phase_c_u = np.mean(U_arr[5000:5500])
if phase_c_u > phase_b_u:
    print("\n  [PASS] Uncertainty increases at reversal.")
else:
    print("\n  [WARN] Reversal uncertainty did not increase as expected.")

print(f"\n  Control signal ranges:")
print(f"    learning_rate_factor : "
      f"{min(alpha_log):.3f} — {max(alpha_log):.3f}")
print(f"    gate_threshold_offset: "
      f"{min(gate_log):.3f} — {max(gate_log):.3f}")
print(f"    exploration_temp     : "
      f"{min(temp_log):.3f} — {max(temp_log):.3f}")
print(f"    risk_aversion        : "
      f"{min(risk_log):.3f} — {max(risk_log):.3f}")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
phase_colors = dict(alpha=0.15, color="gray")

for ax in axes:
    ax.axvspan(0,    200, **phase_colors, label="Phase A (noise)")
    ax.axvspan(200,  500, alpha=0.12, color="steelblue",
               label="Phase B (stable)")
    ax.axvspan(500,  700, alpha=0.12, color="coral",
               label="Phase C (reversal)")

axes[0].plot(t_ms, U_arr, color="crimson",     lw=1.2, label="Ut")
axes[0].plot(t_ms, C_arr, color="steelblue",   lw=1.2, label="Ct")
axes[0].set_title("Uncertainty Ut and confidence Ct")
axes[0].set_ylabel("value")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms, alpha_log, color="purple", lw=1)
axes[1].set_title("Learning rate factor (1 + 2·Ut) → meta-dopamine")
axes[1].set_ylabel("α multiplier")

axes[2].plot(t_ms, temp_log, color="darkorange", lw=1)
axes[2].set_title("Exploration temperature τ (0.5 + 3·Ut)")
axes[2].set_ylabel("temperature")

axes[3].plot(t_ms, gate_log, color="forestgreen", lw=1)
axes[3].set_title("Gate threshold offset (β·Ut - κ·Ct)")
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("threshold offset")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step8_uncertainty.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/step8_uncertainty.png")
print("="*55 + "\n")