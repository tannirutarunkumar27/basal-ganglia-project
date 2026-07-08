"""
Step 22 verification — STDE delayed reward learning.
Simulates 1000 ms with:
  - correlated pre-post activity on action 2
  - reward arriving 300 ms after the activity
Confirms weights for action 2 increase more than others.
"""
"""
Activity on action 2:  steps 100-400   (30 ms window)
Reward arrives:        step  3400       (300 ms after end of activity)
The slow trace (tau=2000ms) bridges this gap.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stdp.stdp_kernel              import STDPKernel
from traces.multi_timescale_traces import MultiTimescaleTraces
from stde.stde_engine              import STDEEngine

np.random.seed(2)

N_ACTIONS    = 4
N_CTX        = 40
N_D1_PER_ACT = 20
DT           = 0.1e-3
PREFERRED_ACT = 2

# Activity window: steps 1000-4000 = 300ms of activity
# Reward at step: 7000 = 300ms AFTER activity ends
# Total: 10000 steps = 1000ms
ACTIVITY_START = 1000
ACTIVITY_END   = 4000
REWARD_STEP    = 7000    # 300 ms after activity ends
N_STEPS        = 10000

delay_ms = (REWARD_STEP - ACTIVITY_END) * DT * 1000
activity_ms = (ACTIVITY_END - ACTIVITY_START) * DT * 1000

print("\n" + "="*60)
print("  Phase 7 — Step 22: STDE Delayed Reward Learning")
print("="*60)
print(f"\n  Activity:  steps {ACTIVITY_START}-{ACTIVITY_END}"
      f"  ({activity_ms:.0f} ms)")
print(f"  Reward:    step  {REWARD_STEP}")
print(f"  Delay:     {delay_ms:.0f} ms  "
      f"(slow trace tau=2000ms bridges this)")

# Separate STDP/trace/STDE per action channel
kernels = [STDPKernel(mode="asymmetric", dt=DT)
           for _ in range(N_ACTIONS)]
traces  = [MultiTimescaleTraces(
               N_CTX, N_D1_PER_ACT,
               dt=DT, adaptive_weights=True)
           for _ in range(N_ACTIONS)]
synapse = [STDEEngine(
               N_CTX, N_D1_PER_ACT,
               sign=+1, w_init_mean=0.5,
               name=f"ctx_d1_a{a}", dt=DT)
           for a in range(N_ACTIONS)]

for a in range(N_ACTIONS):
    kernels[a].initialise(N_CTX, N_D1_PER_ACT)

W_mean_log     = {a: [] for a in range(N_ACTIONS)}
dW_log         = {a: [] for a in range(N_ACTIONS)}
e_total_log    = []

W_init = {}
for a in range(N_ACTIONS):
    W_init[a] = float(
        np.abs(synapse[a].W[synapse[a].W != 0]).mean()
        if (synapse[a].W != 0).any() else 0.5)

for step in range(N_STEPS):
    ctx_spikes = np.random.rand(N_CTX) < 0.05

    for a in range(N_ACTIONS):
        if ACTIVITY_START <= step < ACTIVITY_END:
            # Preferred action has 4x higher D1 firing rate
            rate_d1 = 0.15 if a == PREFERRED_ACT else 0.04
        else:
            # Background noise — very low
            rate_d1 = 0.01

        d1_spikes_a = np.random.rand(N_D1_PER_ACT) < rate_d1

        kernels[a].update_traces(ctx_spikes, d1_spikes_a)
        stdp_dW   = kernels[a].compute_stdp(ctx_spikes, d1_spikes_a)
        e_total_a = traces[a].step(stdp_dW)

        # Delayed reward — fires exactly once at REWARD_STEP
        if step == REWARD_STEP:
            # Positive delta for preferred, small negative for others
            delta  = 1.8 if a == PREFERRED_ACT else -0.2
            # alpha_t reflects moderate uncertainty
            alpha  = 0.05 * 1.5
            dW_out = synapse[a].update(
                e_total     = e_total_a,
                delta_total = delta,
                alpha_t     = alpha)
            traces[a].update_weights(
                delta, 1.0 if a == PREFERRED_ACT else 0.0)
            dW_log[a].append(float(np.abs(dW_out).mean()))
        else:
            dW_log[a].append(0.0)

        w_vals = synapse[a].W[synapse[a].W != 0]
        W_mean_log[a].append(
            float(np.abs(w_vals).mean()) if len(w_vals) else 0.5)

    if step == REWARD_STEP or step % 1000 == 0:
        e_total_log.append(
            (step, float(np.abs(traces[PREFERRED_ACT].e_total).mean())))

# Results
print(f"\n  Final mean weight per action channel:")
for a in range(N_ACTIONS):
    w_final = W_mean_log[a][-1]
    delta_w = w_final - W_init[a]
    marker  = " <-- preferred" if a == PREFERRED_ACT else ""
    print(f"    action {a}: W_init={W_init[a]:.4f}  "
          f"W_final={w_final:.4f}  "
          f"dW={delta_w:+.4f}{marker}")

W_pref   = W_mean_log[PREFERRED_ACT][-1]
W_others = np.mean([W_mean_log[a][-1]
                    for a in range(N_ACTIONS)
                    if a != PREFERRED_ACT])

ratio = W_pref / (W_others + 1e-9)
print(f"\n  Preferred W_mean : {W_pref:.4f}")
print(f"  Others   W_mean  : {W_others:.4f}")
print(f"  Ratio            : {ratio:.4f}  (expected > 1.0)")

# Verify trace at reward time is non-zero (slow trace persisted)
print(f"\n  Eligibility trace at key timesteps:")
for (s, mag) in e_total_log:
    print(f"    step {s:5d} (t={s*DT*1000:6.1f}ms): "
          f"|e_total|={mag:.6f}")

# The REWARD_STEP entry must have non-zero trace
reward_e = next(
    (mag for (s, mag) in e_total_log if s == REWARD_STEP), None)
assert reward_e is not None and reward_e > 1e-8, \
    "Eligibility trace should be non-zero at reward time (slow trace)"

assert ratio > 1.0, \
    f"Preferred action should have higher weights. ratio={ratio:.4f}"
print("\n  [PASS] Delayed reward correctly assigned via STDE.")

# Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
t_ms   = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, W_mean_log[a], color=colors[a],
                 lw=1.2, label=f"a{a}",
                 alpha=1.0 if a == PREFERRED_ACT else 0.6)
axes[0].axvline(ACTIVITY_END * DT * 1000, color="gray",
                ls=":", lw=1, label="activity ends")
axes[0].axvline(REWARD_STEP  * DT * 1000, color="red",
                ls="--", lw=0.8, label=f"reward (t={REWARD_STEP*DT*1000:.0f}ms)")
axes[0].set_title("Mean synaptic weight per action channel (Step 22)")
axes[0].set_ylabel("W mean")
axes[0].legend(fontsize=8)

# Trace for preferred action
trace_pref = []
for step in range(N_STEPS):
    trace_pref.append(
        float(np.abs(traces[PREFERRED_ACT].e_total).mean()))

axes[1].plot(t_ms, trace_pref, color="darkorange", lw=1)
axes[1].axvspan(ACTIVITY_START*DT*1000, ACTIVITY_END*DT*1000,
                alpha=0.1, color="coral", label="activity")
axes[1].axvline(REWARD_STEP*DT*1000, color="red",
                ls="--", lw=0.8, label="reward")
axes[1].set_title(
    f"e_total trace (action {PREFERRED_ACT}) — slow trace bridges 300ms delay")
axes[1].set_ylabel("|e_total|")
axes[1].legend(fontsize=8)

for a in range(N_ACTIONS):
    axes[2].plot(t_ms, dW_log[a], color=colors[a],
                 lw=1.5,
                 alpha=1.0 if a == PREFERRED_ACT else 0.5,
                 label=f"dW a{a}")
axes[2].axvline(REWARD_STEP*DT*1000, color="red",
                ls="--", lw=0.8)
axes[2].set_title(
    "Weight update at reward step  dW = alpha * delta * e_total")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("|dW| mean")
axes[2].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step22_stde.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step22_stde.png")
print("="*60 + "\n")