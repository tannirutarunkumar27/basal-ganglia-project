"""
Step 20 verification — STDP kernel and single eligibility trace.
Simulates correlated pre-post firing and checks the trace
accumulates correctly, then decays when activity stops.
"""
"""
N_STEPS extended so silent phase is long enough for decay.
tau_e = 100ms, need > 300ms silence for ratio < 0.10
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stdp.stdp_kernel         import STDPKernel
from traces.eligibility_trace import EligibilityTrace

np.random.seed(0)

N_PRE  = 20
N_POST = 10
DT     = 0.1e-3
TAU_E  = 100e-3      # 100 ms

# Need silent phase >> tau_e for proper decay test.
# Silent phase = N_STEPS - CORR_END steps.
# For ratio < 0.10:  exp(-t/tau) < 0.10  -> t > tau*ln(10) = 230ms
# Use 400 ms silence = 4000 steps to get ratio ~ exp(-400/100) = 0.018
CORR_END = 1000     # 100 ms correlated activity
N_STEPS  = 6000     # 600 ms total (400 ms silent)

kernel = STDPKernel(A_plus=0.01, A_minus=0.0105,
                     tau_plus=20e-3, tau_minus=20e-3,
                     mode="asymmetric", dt=DT)
trace  = EligibilityTrace(N_PRE, N_POST,
                           tau_e=TAU_E, dt=DT, name="e_single")
kernel.initialise(N_PRE, N_POST)

trace_mag_log = []
stdp_log      = []

print("\n" + "="*55)
print("  Phase 7 — Step 20: STDP Kernel + Eligibility Trace")
print("="*55)
print(f"\n  tau_e        = {TAU_E*1000:.0f} ms")
print(f"  Active phase = {CORR_END * DT * 1000:.0f} ms")
print(f"  Silent phase = {(N_STEPS-CORR_END) * DT * 1000:.0f} ms")
print(f"  Expected decay ratio < exp(-{(N_STEPS-CORR_END)*DT*1000:.0f}"
      f"/{TAU_E*1000:.0f}) = "
      f"{np.exp(-(N_STEPS-CORR_END)*DT/TAU_E):.4f}")

for step in range(N_STEPS):
    if step < CORR_END:
        pre_spikes  = np.random.rand(N_PRE)  < 0.08
        post_spikes = np.random.rand(N_POST) < 0.06
    else:
        # Truly silent — no spikes at all
        pre_spikes  = np.zeros(N_PRE,  dtype=bool)
        post_spikes = np.zeros(N_POST, dtype=bool)

    kernel.update_traces(pre_spikes, post_spikes)
    stdp_dW = kernel.compute_stdp(pre_spikes, post_spikes)
    trace.step(stdp_dW)

    trace_mag_log.append(trace.mean_magnitude())
    stdp_log.append(float(np.abs(stdp_dW).mean()))

peak   = max(trace_mag_log[:CORR_END])
at_end = trace_mag_log[-1]
ratio  = at_end / (peak + 1e-12)
decay_ok = ratio < 0.10

print(f"\n  Trace peak (active phase):  {peak:.6f}")
print(f"  Trace at end (silent phase): {at_end:.6f}")
print(f"  Decay ratio: {ratio:.4f}  (expected < 0.10)")

assert decay_ok, (
    f"Trace should decay to < 10% of peak during silent phase. "
    f"Got ratio={ratio:.4f}. Increase N_STEPS or reduce tau_e.")
print("  [PASS] Eligibility trace accumulates and decays correctly.")

summary = trace.trace_summary()
print(f"\n  Trace summary:")
for k, v in summary.items():
    print(f"    {k:<18s}: {v}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t_ms = np.arange(N_STEPS) * DT * 1000

axes[0].plot(t_ms, stdp_log, color="steelblue",
             lw=0.8, alpha=0.8)
axes[0].axvline(CORR_END * DT * 1000, color="red",
                ls="--", lw=0.8, label="activity stops")
axes[0].set_title("STDP update magnitude per step (Step 20)")
axes[0].set_ylabel("|dW| mean")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, N_STEPS * DT * 1000)

axes[1].plot(t_ms, trace_mag_log, color="darkorange", lw=1.2)
axes[1].axvline(CORR_END * DT * 1000, color="red",
                ls="--", lw=0.8, label="activity stops")
# Mark the 10% threshold
axes[1].axhline(peak * 0.10, color="gray", ls=":",
                lw=0.8, label="10% of peak")
axes[1].set_title(
    "Eligibility trace — accumulates then exponentially decays")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("|e| mean")
axes[1].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step20_trace.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step20_trace.png")
print("="*55 + "\n")