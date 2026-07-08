"""
Step 4 verification:
  - builds full BGNetwork
  - validates topology, sparsity, weights
  - runs 500 ms simulation
  - calibrates spike rates
  - estimates energy budget
  - saves plots to results/
"""

import sys, os
# Make sure phase2/ is on the path regardless of working directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.bg_network                import BGNetwork
from calibration.topology_validator   import (
    validate_topology, validate_sparsity, validate_weight_magnitudes)
from calibration.spike_rate_calibrator import calibrate_all
from calibration.energy_estimator      import (
    estimate_energy, print_energy_report)

DT       = 0.1e-3      # 0.1 ms
SIM_TIME = 0.5         # 500 ms
N_STEPS  = int(SIM_TIME / DT)

print("\n" + "=" * 60)
print("  Phase 2 — Step 4: Connectivity & Calibration")
print("=" * 60)

# ── Build network ──────────────────────────────────────────────
try:
    net = BGNetwork(dt=DT)
except Exception as e:
    print(f"\n  [FATAL] BGNetwork build failed: {e}")
    raise

# ── Topology validation ────────────────────────────────────────
validate_topology(net.cons)
validate_sparsity(net.cons)
validate_weight_magnitudes(net.cons)

# ── Spike-rate calibration (isolated populations) ──────────────
calibrate_all(dt=DT)

# ── 500 ms network simulation ──────────────────────────────────
print(f"\n  Running {SIM_TIME * 1000:.0f} ms network simulation...")
net.reset()

pop_names  = list(net.pops.keys())
spike_hist = {n: [] for n in pop_names}
gpi_rates  = []

for step in range(N_STEPS):
    t = step * DT
    # 5 Hz sinusoidal cortex drive (simulates task rhythm)
    ctx_amplitude = 0.5e-9 * (np.sin(2 * np.pi * 5 * t) + 1.0)
    ctx_in = np.full(net.pops["cortex"].N, ctx_amplitude)

    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    for n in pop_names:
        spike_hist[n].append(int(spks[n].sum()))

    gpi_rates.append(net.pops["gpi"].population_rate(50))

# ── Energy estimate ────────────────────────────────────────────
energy = estimate_energy(net.pops, N_STEPS, DT)
print_energy_report(energy)

# ── Final firing rates ─────────────────────────────────────────
print("\n--- Final Firing Rates ---")
for name, pop in net.pops.items():
    rate = pop.population_rate(window_steps=1000)
    print(f"  {name:<24s}  {rate:6.1f} Hz")

# ── Plots ──────────────────────────────────────────────────────
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
t_axis = np.arange(N_STEPS) * DT * 1000   # ms

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# --- Pathway activity ---
pathway_colors = {
    "d1_msn" : "steelblue",
    "d2_msn" : "coral",
    "stn"    : "forestgreen",
    "gpi"    : "darkorange",
}
smooth = np.ones(100) / 100
for pname, color in pathway_colors.items():
    hist = np.array(spike_hist[pname], dtype=float)
    axes[0].plot(t_axis,
                 np.convolve(hist, smooth, mode="same"),
                 label=pname, color=color, lw=1.2)
axes[0].set_title("Core pathway activity (smoothed spike count per step)")
axes[0].set_ylabel("Spikes / step")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_xlim(0, SIM_TIME * 1000)

# --- GPi gate signal ---
axes[1].plot(t_axis, gpi_rates, color="darkorange", lw=1)
axes[1].axhline(y=30, color="red", ls="--", lw=0.8,
                label="Action release threshold")
axes[1].set_title("GPi population rate — action gate signal")
axes[1].set_ylabel("Rate (Hz)")
axes[1].legend(fontsize=8)
axes[1].set_xlim(0, SIM_TIME * 1000)

# --- Neuromodulators ---
nm_colors = {
    "snc"            : "gold",
    "serotonin"      : "mediumpurple",
    "norepinephrine" : "teal",
}
smooth2 = np.ones(200) / 200
for pname, color in nm_colors.items():
    hist = np.array(spike_hist[pname], dtype=float)
    axes[2].plot(t_axis,
                 np.convolve(hist, smooth2, mode="same"),
                 label=pname, color=color, lw=1.2)
axes[2].set_title("Neuromodulator populations")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Spikes / step")
axes[2].legend(fontsize=8)
axes[2].set_xlim(0, SIM_TIME * 1000)

plt.tight_layout()
out_path = os.path.join(HERE, "results", "step4_network.png")
plt.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close()

print(f"\n  Network plot saved: {out_path}")
print("=" * 60 + "\n")