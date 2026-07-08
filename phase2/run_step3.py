"""
Step 3 verification: simulate all populations for 500 ms
with constant input current, check spike rates and dynamics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurons.population_factory import build_all_populations
from neurons.population_params  import POPULATION_PARAMS

DT         = 0.1e-3      # 0.1 ms timestep
SIM_TIME   = 0.5         # 500 ms
N_STEPS    = int(SIM_TIME / DT)

def calibration_current(target_hz: float, params: dict,
                         dt: float = DT) -> float:
    """
    Rough analytical estimate of DC current needed to drive
    a leaky integrator at target_hz (used for initial calibration).
    """
    gL = params.get("gL", 10e-9)
    EL = params.get("EL", -70e-3)
    VT = params.get("VT", -50e-3)
    # Rheobase approximation: I ~ gL*(VT - EL) + offset
    rheobase = gL * (VT - EL)
    scale    = 1.0 + target_hz / 100.0
    return rheobase * scale

print("\n" + "="*60)
print("  Phase 2 — Step 3: AdEx Population Verification")
print("="*60)

pops = build_all_populations(dt=DT)

print("\n  Simulating 500 ms with calibration currents...\n")

results = {}
for name, pop in pops.items():
    params     = POPULATION_PARAMS[name]
    I_cal      = calibration_current(params["target_rate_hz"], params)
    I_input    = np.full(pop.N, I_cal)
    pop.reset_state()

    for step in range(N_STEPS):
        t = step * DT
        # small Poisson noise to avoid synchrony
        I_noisy = I_input + np.random.normal(0, I_cal * 0.1, pop.N)
        pop.step(I_noisy, t)

    rate = pop.population_rate(window_steps=N_STEPS)
    target = params["target_rate_hz"]
    error  = abs(rate - target) / max(target, 1e-9) * 100

    results[name] = {"rate": rate, "target": target, "error": error}
    status = "OK" if error < 50 else "WARN"
    print(f"  [{status}] {name:<22s} "
          f"rate={rate:6.1f} Hz  target={target:5.1f} Hz  "
          f"err={error:5.1f}%  spikes={pop.total_spikes:6d}")

# Save raster plot
fig, axes = plt.subplots(len(pops), 1, figsize=(14, 20), sharex=True)
for ax, (name, pop) in zip(axes, pops.items()):
    if len(pop.spike_log) > 0:
        spike_arr = np.array(pop.spike_log, dtype=float)
        T_plot = min(2000, spike_arr.shape[0])
        for neuron_i in range(min(pop.N, 10)):
            spike_times = np.where(spike_arr[:T_plot, neuron_i])[0] * DT * 1000
            ax.scatter(spike_times,
                       np.full_like(spike_times, neuron_i),
                       s=1, color="steelblue", alpha=0.7)
    ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    ax.set_yticks([])

axes[-1].set_xlabel("Time (ms)")
fig.suptitle("AdEx Raster — first 200 ms, first 10 neurons per pop", y=1.01)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/step3_raster.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n  Raster saved: results/step3_raster.png")
print("="*60 + "\n")