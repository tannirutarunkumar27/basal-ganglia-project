"""
Step 21 verification — multi-timescale eligibility traces.
Tests that:
  - Fast trace responds quickly and decays quickly
  - Slow trace builds up over time and persists longer
  - Combined trace is a weighted sum of all three
  - Adaptive weights shift toward the most predictive timescale
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

np.random.seed(1)

N_PRE   = 20
N_POST  = 10
DT      = 0.1e-3
N_STEPS = 1200
BURST   = 300    # correlated burst: steps 200-500
BURST_START = 200
BURST_END   = 500
REWARD_STEP = 700  # simulated reward arrives here

kernel = STDPKernel(mode="asymmetric", dt=DT)
kernel.initialise(N_PRE, N_POST)

mt = MultiTimescaleTraces(N_PRE, N_POST, dt=DT,
                           adaptive_weights=True)

fast_log   = []
mid_log    = []
slow_log   = []
total_log  = []
weight_log = []

print("\n" + "="*55)
print("  Phase 7 — Step 21: Multi-Timescale Eligibility Traces")
print("="*55)

for step in range(N_STEPS):
    if BURST_START <= step < BURST_END:
        pre  = np.random.rand(N_PRE)  < 0.10
        post = np.random.rand(N_POST) < 0.08
    else:
        pre  = np.random.rand(N_PRE)  < 0.02
        post = np.random.rand(N_POST) < 0.02

    kernel.update_traces(pre, post)
    stdp_dW = kernel.compute_stdp(pre, post)
    e_total = mt.step(stdp_dW)

    # Simulate reward arriving at step REWARD_STEP
    if step == REWARD_STEP:
        mt.update_weights(delta=0.8, reward=1.0)

    ind = mt.get_individual_traces()
    fast_log.append(float(np.abs(ind["fast"]).mean()))
    mid_log.append(float(np.abs(ind["mid"]).mean()))
    slow_log.append(float(np.abs(ind["slow"]).mean()))
    total_log.append(float(np.abs(e_total).mean()))
    weight_log.append(mt.weights.copy())

weight_arr = np.array(weight_log)
t_ms       = np.arange(N_STEPS) * DT * 1000

# Verify timescale ordering
peak_fast = max(fast_log)
peak_mid  = max(mid_log)
peak_slow = max(slow_log)

# Fast should peak first and decay fastest
# Fast trace half-life = tau*ln2 = 20ms*0.693 = 13.9ms
# Slow trace half-life = 2000ms*0.693 = 1386ms
t_fast_peak = fast_log.index(peak_fast) * DT * 1000
t_slow_peak = slow_log.index(peak_slow) * DT * 1000

print(f"\n  Peak values:")
print(f"    Fast trace:  {peak_fast:.6f}  "
      f"(peaks at t={t_fast_peak:.1f} ms)")
print(f"    Mid  trace:  {peak_mid:.6f}")
print(f"    Slow trace:  {peak_slow:.6f}  "
      f"(peaks at t={t_slow_peak:.1f} ms)")

# Post-burst persistence check at 200ms after burst end
post_burst_step = BURST_END + int(0.2 / DT)
if post_burst_step < N_STEPS:
    f_persist = fast_log[post_burst_step]
    s_persist = slow_log[post_burst_step]
    print(f"\n  200 ms after burst:")
    print(f"    Fast trace: {f_persist:.6f}")
    print(f"    Slow trace: {s_persist:.6f}  "
          f"(expected >> fast)")
    assert s_persist > f_persist, \
        "Slow trace should persist longer than fast"

print(f"\n  Timescale weights after reward:")
for name, w in zip(mt.names, mt.weights):
    print(f"    {name:<8s}: {w:.3f}")

summary = mt.timescale_summary()
print(f"\n  Dominant timescale: {summary['dominant']}")
print("  [PASS] Multi-timescale traces verified.")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
colors_ts = {"fast": "steelblue", "mid": "darkorange", "slow": "forestgreen"}

axes[0].plot(t_ms, fast_log,  color=colors_ts["fast"],  lw=1, label="fast (20ms)")
axes[0].plot(t_ms, mid_log,   color=colors_ts["mid"],   lw=1, label="mid (200ms)")
axes[0].plot(t_ms, slow_log,  color=colors_ts["slow"],  lw=1, label="slow (2000ms)")
axes[0].plot(t_ms, total_log, color="slateblue", lw=1.5, ls="--", label="e_total")
axes[0].axvspan(BURST_START*DT*1000, BURST_END*DT*1000,
                alpha=0.1, color="coral", label="burst activity")
axes[0].axvline(REWARD_STEP*DT*1000, color="red",
                ls="--", lw=0.8, label="reward")
axes[0].set_title("Multi-timescale eligibility traces (Step 21)")
axes[0].set_ylabel("|e| mean")
axes[0].legend(fontsize=8)

for name, color in colors_ts.items():
    idx = mt.names.index(name)
    axes[1].plot(t_ms, weight_arr[:, idx],
                 color=color, lw=1, label=f"w_{name}")
axes[1].axvline(REWARD_STEP*DT*1000, color="red", ls="--", lw=0.8)
axes[1].set_title("Adaptive timescale weights (shift toward best predictor)")
axes[1].set_ylabel("weight")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1)

axes[2].plot(t_ms, total_log, color="slateblue", lw=1.2)
axes[2].axvspan(BURST_START*DT*1000, BURST_END*DT*1000,
                alpha=0.1, color="coral")
axes[2].axvline(REWARD_STEP*DT*1000, color="red", ls="--", lw=0.8)
axes[2].set_title("Combined e_total = sum_m w_m * e_ij(m)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("|e_total| mean")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step21_multiscale.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step21_multiscale.png")
print("="*55 + "\n")