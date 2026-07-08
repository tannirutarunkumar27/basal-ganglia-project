"""
Spike-rate calibration:
  Runs each population in isolation and adjusts tonic
  drive until firing rate matches biological target.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neurons.adex_population   import AdExPopulation
from neurons.population_params import POPULATION_PARAMS

def calibrate_population(name: str,
                          target_hz: float,
                          dt: float = 0.1e-3,
                          sim_time: float = 1.0,
                          tol_pct: float = 20.0,
                          max_iter: int = 20) -> float:
    """
    Binary-search for the tonic current that achieves target_hz.
    Returns the calibrated current value.
    """
    params  = POPULATION_PARAMS[name]
    N_steps = int(sim_time / dt)

    # Initial bracket
    I_lo, I_hi = 0.0, 5e-9

    for iteration in range(max_iter):
        I_mid = (I_lo + I_hi) / 2.0
        pop   = AdExPopulation(N=20, params=params, dt=dt, name=name)
        pop.reset_state()

        for step in range(N_steps):
            t = step * dt
            I = np.full(pop.N, I_mid) + \
                np.random.normal(0, I_mid * 0.05, pop.N)
            pop.step(I, t)

        rate = pop.population_rate(window_steps=N_steps)
        err  = abs(rate - target_hz) / max(target_hz, 1e-9) * 100

        if err < tol_pct:
            break
        if rate < target_hz:
            I_lo = I_mid
        else:
            I_hi = I_mid

    return I_mid, rate, err


def calibrate_all(dt: float = 0.1e-3) -> dict:
    print("\n--- Spike-Rate Calibration ---")
    calibrated = {}
    for name, params in POPULATION_PARAMS.items():
        target = params["target_rate_hz"]
        I_cal, rate, err = calibrate_population(
            name, target, dt=dt)
        calibrated[name] = I_cal
        status = "OK " if err < 30 else "WARN"
        print(f"  [{status}] {name:<22s} "
              f"target={target:5.1f} Hz  "
              f"achieved={rate:5.1f} Hz  "
              f"I_cal={I_cal:.3e} A")
    return calibrated