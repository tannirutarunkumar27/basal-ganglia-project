"""
Energy budget estimator for neuromorphic deployment.
Estimates total synaptic events and spike energy per episode.
"""

import numpy as np

# Approximate energy per spike on neuromorphic hardware (Intel Loihi)
ENERGY_PER_SPIKE_J = 23e-12    # 23 pJ per spike

def estimate_energy(pops: dict,
                    episode_steps: int,
                    dt: float = 0.1e-3) -> dict:
    """
    Estimates energy consumption from spike counts.
    Uses total_spikes logged in each AdExPopulation.
    """
    total_spikes = 0
    report = {}

    for name, pop in pops.items():
        n_spikes = pop.total_spikes
        energy_J = n_spikes * ENERGY_PER_SPIKE_J
        rate_hz  = n_spikes / (episode_steps * dt + 1e-12) / pop.N
        report[name] = {
            "n_spikes" : n_spikes,
            "energy_pJ": energy_J * 1e12,
            "rate_hz"  : rate_hz
        }
        total_spikes += n_spikes

    total_energy_nJ = total_spikes * ENERGY_PER_SPIKE_J * 1e9
    report["TOTAL"] = {
        "n_spikes"  : total_spikes,
        "energy_nJ" : total_energy_nJ,
    }
    return report


def print_energy_report(report: dict):
    print("\n--- Energy Budget Estimate ---")
    for name, data in report.items():
        if name == "TOTAL":
            print(f"\n  TOTAL  spikes={data['n_spikes']:8d}  "
                  f"energy={data['energy_nJ']:.2f} nJ")
        else:
            print(f"  {name:<22s}  "
                  f"spikes={data['n_spikes']:6d}  "
                  f"energy={data['energy_pJ']:6.1f} pJ  "
                  f"rate={data['rate_hz']:5.1f} Hz")