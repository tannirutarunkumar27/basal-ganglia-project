"""
PopulationFactory: builds all AdEx populations from the preset table.
Returns a dict keyed by population name.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neurons.adex_population  import AdExPopulation
from neurons.population_params import POPULATION_PARAMS

def build_all_populations(dt: float = 0.1e-3,
                          param_noise: float = 0.05) -> dict:
    """
    Instantiates all 12 AdEx populations.
    Returns: { population_name: AdExPopulation }
    """
    populations = {}
    for name, params in POPULATION_PARAMS.items():
        N = params.get("n_neurons", 40)
        populations[name] = AdExPopulation(
            N          = N,
            params     = params,
            dt         = dt,
            param_noise= param_noise,
            name       = name
        )
        print(f"  [BUILD] {name:<20s}  N={N:3d}  "
              f"target={params['target_rate_hz']:5.1f} Hz  "
              f"sign={'+' if params['sign']>0 else '-'}")
    return populations