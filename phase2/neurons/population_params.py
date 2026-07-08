"""
Biologically grounded AdEx parameter presets for each
neuron population in the BG-cortex-thalamus circuit.

References:
  Brette & Gerstner (2005) — AdEx model
  Izhikevich (2007)        — neuron type classification
  Bogacz & Gurney (2007)   — BG parameters
"""

POPULATION_PARAMS = {

    # ── Cortex: regular spiking pyramidal neurons ──────────────────────
    "cortex": {
        "C": 200e-12, "gL": 10e-9, "EL": -70e-3,
        "VT": -50e-3, "dT": 2e-3,
        "Vreset": -58e-3, "tau_w": 100e-3,
        "a": 2e-9, "b": 100e-12,       # moderate spike adaptation
        "target_rate_hz": 5.0,
        "n_neurons": 100,
        "sign": +1,                     # excitatory
        "description": "Regular-spiking pyramidal, drives BG and STN"
    },

    # ── D1 MSNs: direct pathway, dopamine excites (Go) ─────────────────
    "d1_msn": {
        "C": 150e-12, "gL": 8e-9, "EL": -80e-3,
        "VT": -55e-3, "dT": 1.5e-3,
        "Vreset": -70e-3, "tau_w": 200e-3,
        "a": 0.5e-9, "b": 50e-12,      # low adaptation — tonic inhibition
        "target_rate_hz": 2.0,
        "n_neurons": 80,
        "sign": -1,                     # inhibitory output to GPi
        "description": "Striatal D1, direct pathway Go cells"
    },

    # ── D2 MSNs: indirect pathway, dopamine inhibits (No-Go) ───────────
    "d2_msn": {
        "C": 150e-12, "gL": 8e-9, "EL": -80e-3,
        "VT": -55e-3, "dT": 1.5e-3,
        "Vreset": -70e-3, "tau_w": 200e-3,
        "a": 0.5e-9, "b": 50e-12,
        "target_rate_hz": 2.0,
        "n_neurons": 80,
        "sign": -1,                     # inhibitory output to GPe
        "description": "Striatal D2, indirect pathway No-Go cells"
    },

    # ── STN: hyperdirect pathway, fast conflict signal ──────────────────
    "stn": {
        "C": 60e-12, "gL": 5e-9, "EL": -60e-3,
        "VT": -45e-3, "dT": 3e-3,
        "Vreset": -65e-3, "tau_w": 80e-3,
        "a": 0.5e-9, "b": 200e-12,     # strong burst-then-adapt
        "target_rate_hz": 20.0,         # tonically active
        "n_neurons": 40,
        "sign": +1,                     # excitatory output to GPi/GPe
        "description": "Subthalamic nucleus, hyperdirect conflict brake"
    },

    # ── GPe: indirect pathway relay ─────────────────────────────────────
    "gpe": {
        "C": 60e-12, "gL": 5e-9, "EL": -65e-3,
        "VT": -48e-3, "dT": 2e-3,
        "Vreset": -60e-3, "tau_w": 60e-3,
        "a": 1e-9, "b": 100e-12,
        "target_rate_hz": 50.0,         # tonically active, high rate
        "n_neurons": 40,
        "sign": -1,                     # inhibitory output to STN/GPi
        "description": "Globus pallidus externa, indirect relay"
    },

    # ── GPi: output nucleus, gate to thalamus ────────────────────────────
    "gpi": {
        "C": 60e-12, "gL": 5e-9, "EL": -65e-3,
        "VT": -48e-3, "dT": 2e-3,
        "Vreset": -60e-3, "tau_w": 60e-3,
        "a": 1e-9, "b": 100e-12,
        "target_rate_hz": 60.0,         # high tonic rate to suppress thalamus
        "n_neurons": 40,
        "sign": -1,                     # inhibitory gate to thalamus
        "description": "Globus pallidus interna, BG output gate"
    },

    # ── Thalamus: relay to motor cortex ─────────────────────────────────
    "thalamus": {
        "C": 100e-12, "gL": 10e-9, "EL": -70e-3,
        "VT": -50e-3, "dT": 2e-3,
        "Vreset": -65e-3, "tau_w": 100e-3,
        "a": 1e-9, "b": 50e-12,
        "target_rate_hz": 10.0,
        "n_neurons": 40,
        "sign": +1,                     # excitatory to motor cortex
        "description": "Thalamic relay, disinhibited when GPi silent"
    },

    # ── SNc: dopamine neurons, reward prediction ─────────────────────────
    "snc": {
        "C": 40e-12, "gL": 4e-9, "EL": -55e-3,
        "VT": -45e-3, "dT": 3e-3,
        "Vreset": -60e-3, "tau_w": 500e-3,   # very slow adaptation
        "a": 0.2e-9, "b": 20e-12,
        "target_rate_hz": 4.0,           # low tonic dopamine
        "n_neurons": 20,
        "sign": +1,                      # dopaminergic modulation
        "description": "SNc dopamine neurons, reward / RPE signalling"
    },

    # ── Serotonin / 5-HT: risk and aversion control ──────────────────────
    "serotonin": {
        "C": 40e-12, "gL": 3e-9, "EL": -60e-3,
        "VT": -45e-3, "dT": 2e-3,
        "Vreset": -65e-3, "tau_w": 800e-3,   # slow neuromodulator
        "a": 0.1e-9, "b": 10e-12,
        "target_rate_hz": 2.0,
        "n_neurons": 20,
        "sign": +1,
        "description": "Raphe serotonin units, risk/aversion neuromod"
    },

    # ── Norepinephrine (NE): arousal and uncertainty ──────────────────────
    "norepinephrine": {
        "C": 40e-12, "gL": 3e-9, "EL": -60e-3,
        "VT": -45e-3, "dT": 2e-3,
        "Vreset": -65e-3, "tau_w": 600e-3,
        "a": 0.1e-9, "b": 10e-12,
        "target_rate_hz": 2.0,
        "n_neurons": 20,
        "sign": +1,
        "description": "LC norepinephrine units, arousal/surprise signal"
    },

    # ── Bayesian belief neurons ───────────────────────────────────────────
    "bayesian_layer": {
        "C": 100e-12, "gL": 8e-9, "EL": -68e-3,
        "VT": -50e-3, "dT": 2e-3,
        "Vreset": -60e-3, "tau_w": 150e-3,
        "a": 1.5e-9, "b": 60e-12,
        "target_rate_hz": 10.0,
        "n_neurons": 60,
        "sign": +1,
        "description": "Encodes posterior belief per action"
    },

    # ── Reasoning / neuro-symbolic layer ─────────────────────────────────
    "reasoning_layer": {
        "C": 120e-12, "gL": 9e-9, "EL": -68e-3,
        "VT": -50e-3, "dT": 2e-3,
        "Vreset": -60e-3, "tau_w": 180e-3,
        "a": 2e-9, "b": 80e-12,
        "target_rate_hz": 8.0,
        "n_neurons": 60,
        "sign": +1,
        "description": "Neuro-symbolic inference and explanation units"
    },
}