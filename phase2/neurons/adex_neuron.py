"""
Adaptive Exponential Integrate-and-Fire (AdEx) Neuron
------------------------------------------------------
C dV/dt = -gL(V-EL) + gL*dT*exp((V-VT)/dT) - w + I(t)
tau_w dw/dt = a(V-EL) - w

Spike/reset:
  if V >= V_spike:  emit spike, V -> V_reset, w -> w + b
"""

import numpy as np

class AdExNeuron:
    """
    Single AdEx neuron with all biophysical parameters.
    Supports Euler integration at fixed dt.
    """

    # Default biophysical parameters (cortical pyramidal neuron baseline)
    DEFAULT_PARAMS = {
        "C"       : 200e-12,   # membrane capacitance (F)
        "gL"      : 10e-9,     # leak conductance (S)
        "EL"      : -70e-3,    # leak reversal potential (V)
        "VT"      : -50e-3,    # threshold slope factor (V)
        "dT"      : 2e-3,      # slope factor delta_T (V)
        "Vspike"  : 20e-3,     # spike detection threshold (V)
        "Vreset"  : -58e-3,    # reset potential (V)
        "tau_w"   : 100e-3,    # adaptation time constant (s)
        "a"       : 2e-9,      # subthreshold adaptation (S)
        "b"       : 0.0e-12,   # spike-triggered adaptation (A)
        "Vmax"    : 40e-3,     # hard clamp to prevent overflow
    }

    def __init__(self, params: dict = None, dt: float = 0.1e-3):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.C      = p["C"]
        self.gL     = p["gL"]
        self.EL     = p["EL"]
        self.VT     = p["VT"]
        self.dT     = p["dT"]
        self.Vspike = p["Vspike"]
        self.Vreset = p["Vreset"]
        self.tau_w  = p["tau_w"]
        self.a      = p["a"]
        self.b      = p["b"]
        self.Vmax   = p["Vmax"]
        self.dt     = dt

        # State
        self.V      = self.EL
        self.w      = 0.0
        self.spike  = False
        self.t_last_spike = -np.inf
        self.refractory_period = 2e-3   # 2 ms absolute refractory

    def reset_state(self):
        self.V     = self.EL
        self.w     = 0.0
        self.spike = False
        self.t_last_spike = -np.inf

    def step(self, I_ext: float, t: float) -> bool:
        """
        Integrate one time step.
        Returns True if a spike was emitted this step.
        """
        self.spike = False

        # Absolute refractory period
        if (t - self.t_last_spike) < self.refractory_period:
            self.V = self.Vreset
            return False

        # dV/dt = [-gL(V-EL) + gL*dT*exp((V-VT)/dT) - w + I] / C
        exp_term = self.gL * self.dT * np.exp(
            np.clip((self.V - self.VT) / self.dT, -10, 10)
        )
        dV = (-self.gL * (self.V - self.EL)
              + exp_term - self.w + I_ext) / self.C

        # dw/dt = [a*(V-EL) - w] / tau_w
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w

        self.V = np.clip(self.V + self.dt * dV, -0.1, self.Vmax)
        self.w += self.dt * dw

        # Spike detection and reset
        if self.V >= self.Vspike:
            self.V = self.Vreset
            self.w += self.b
            self.spike = True
            self.t_last_spike = t

        return self.spike