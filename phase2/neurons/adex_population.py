"""
Vectorised AdEx population for efficient simulation of N neurons.
All operations are numpy-vectorised — no Python loops over neurons.
"""

import numpy as np

class AdExPopulation:
    """
    N AdEx neurons simulated in parallel using numpy arrays.
    Each neuron can have slightly different parameters via noise.
    """

    def __init__(self, N: int, params: dict = None,
                 dt: float = 0.1e-3,
                 param_noise: float = 0.05,
                 name: str = "population"):
        self.N    = N
        self.dt   = dt
        self.name = name

        # Base parameters
        base = {
            "C"      : 200e-12,
            "gL"     : 10e-9,
            "EL"     : -70e-3,
            "VT"     : -50e-3,
            "dT"     : 2e-3,
            "Vspike" : 20e-3,
            "Vreset" : -58e-3,
            "tau_w"  : 100e-3,
            "a"      : 2e-9,
            "b"      : 0.0,
            **(params or {})
        }

        # Add biological variability across neurons
        noise = lambda v: v * (1 + param_noise * np.random.randn(N))

        self.C       = noise(base["C"])
        self.gL      = noise(base["gL"])
        self.EL      = noise(base["EL"])
        self.VT      = noise(base["VT"])
        self.dT      = np.abs(noise(base["dT"]))
        self.Vspike  = np.full(N, base["Vspike"])
        self.Vreset  = np.full(N, base["Vreset"])
        self.tau_w   = np.abs(noise(base["tau_w"]))
        self.a       = np.abs(noise(base["a"]))
        self.b       = np.full(N, base["b"])

        # Refractory
        self.t_ref   = 2e-3
        self.t_last  = np.full(N, -np.inf)

        # State vectors
        self.V       = self.EL.copy()
        self.w       = np.zeros(N)
        self.spikes  = np.zeros(N, dtype=bool)

        # Spike history [timesteps x N]
        self.spike_log = []
        self.total_spikes = 0

    def reset_state(self):
        self.V      = self.EL.copy()
        self.w      = np.zeros(self.N)
        self.spikes = np.zeros(self.N, dtype=bool)
        self.t_last = np.full(self.N, -np.inf)
        self.spike_log.clear()
        self.total_spikes = 0

    def step(self, I_ext: np.ndarray, t: float) -> np.ndarray:
        """
        Advance all N neurons by one dt.
        I_ext: array of shape (N,) — total synaptic + external current.
        Returns boolean spike array of shape (N,).
        """
        self.spikes[:] = False

        # Refractory mask: True = neuron is free to spike
        free = (t - self.t_last) >= self.t_ref

        # AdEx dynamics (only on free neurons)
        exp_arg  = np.clip((self.V - self.VT) / self.dT, -10, 10)
        exp_term = self.gL * self.dT * np.exp(exp_arg)

        dV = ((-self.gL * (self.V - self.EL)
               + exp_term - self.w + I_ext) / self.C)
        dw = ((self.a * (self.V - self.EL) - self.w) / self.tau_w)

        # Update only free neurons
        self.V[free] = np.clip(
            self.V[free] + self.dt * dV[free], -0.1, 0.04
        )
        self.w += self.dt * dw

        # Spike detection
        fired = free & (self.V >= self.Vspike)
        self.V[fired]      = self.Vreset[fired]
        self.w[fired]     += self.b[fired]
        self.t_last[fired] = t
        self.spikes        = fired

        self.spike_log.append(fired.copy())
        self.total_spikes += int(fired.sum())

        return self.spikes

    def mean_firing_rate(self, window_steps: int = 1000) -> np.ndarray:
        """
        Returns per-neuron mean firing rate (Hz) over last window_steps.
        """
        if len(self.spike_log) == 0:
            return np.zeros(self.N)
        recent = np.array(self.spike_log[-window_steps:], dtype=float)
        return recent.mean(axis=0) / self.dt

    def population_rate(self, window_steps: int = 100) -> float:
        """Mean population firing rate in Hz."""
        return float(self.mean_firing_rate(window_steps).mean())