"""
Defines ALL 16 synaptic connections in the BG-cortex-thalamus network.

Pathway            Sign    Plastic    Biological role
──────────────────────────────────────────────────────
cortex → striatum  +1      YES        Corticostriatal (reward learning)
cortex → stn       +1      NO         Hyperdirect pathway
d1_msn → gpi       -1      YES        Direct pathway Go
d2_msn → gpe       -1      YES        Indirect pathway No-Go
gpe → stn          -1      NO         GPe modulates STN
gpe → gpi          -1      NO         GPe inhibits GPi
stn → gpi          +1      NO         STN drives GPi
gpi → thalamus     -1      NO         BG output gate
thalamus → cortex  +1      NO         Thalamocortical relay
snc → striatum     +1      NO         Dopamine modulation
serotonin → gpi    +1      NO         Risk signal
norepinephrine → stn +1    NO         Arousal/uncertainty
bayesian → striatum +1     NO         Belief modulation
bayesian → gpi     +1      NO         Belief modulation on gate
"""

import numpy as np
from connectivity.synapse import SynapseGroup


def build_connectivity(pops: dict, dt: float = 0.1e-3) -> dict:
    """
    Constructs all SynapseGroup objects for the full network.
    Returns a dict keyed by connection name.
    """

    def N(name):
        return pops[name].N

    connections = {}

    # 1. Cortex -> Striatum (D1 + D2)
    connections["ctx_d1"] = SynapseGroup(
        name="ctx_d1", N_pre=N("cortex"), N_post=N("d1_msn"),
        sign=+1, weight_mean=0.8e-9, weight_std=0.15e-9,
        conn_prob=0.4, tau_syn=5e-3, plastic=True, dt=dt)

    connections["ctx_d2"] = SynapseGroup(
        name="ctx_d2", N_pre=N("cortex"), N_post=N("d2_msn"),
        sign=+1, weight_mean=0.8e-9, weight_std=0.15e-9,
        conn_prob=0.4, tau_syn=5e-3, plastic=True, dt=dt)

    # 2. Cortex -> STN (hyperdirect)
    connections["ctx_stn"] = SynapseGroup(
        name="ctx_stn", N_pre=N("cortex"), N_post=N("stn"),
        sign=+1, weight_mean=0.6e-9, weight_std=0.1e-9,
        conn_prob=0.3, tau_syn=3e-3, plastic=False, dt=dt)

    # 3. D1 -> GPi (direct Go)
    connections["d1_gpi"] = SynapseGroup(
        name="d1_gpi", N_pre=N("d1_msn"), N_post=N("gpi"),
        sign=-1, weight_mean=1.2e-9, weight_std=0.2e-9,
        conn_prob=0.5, tau_syn=6e-3, plastic=True, dt=dt)

    # 4. D2 -> GPe (indirect No-Go)
    connections["d2_gpe"] = SynapseGroup(
        name="d2_gpe", N_pre=N("d2_msn"), N_post=N("gpe"),
        sign=-1, weight_mean=1.2e-9, weight_std=0.2e-9,
        conn_prob=0.5, tau_syn=6e-3, plastic=True, dt=dt)

    # 5. GPe -> STN
    connections["gpe_stn"] = SynapseGroup(
        name="gpe_stn", N_pre=N("gpe"), N_post=N("stn"),
        sign=-1, weight_mean=1.0e-9, weight_std=0.15e-9,
        conn_prob=0.6, tau_syn=8e-3, plastic=False, dt=dt)

    # 6. GPe -> GPi
    connections["gpe_gpi"] = SynapseGroup(
        name="gpe_gpi", N_pre=N("gpe"), N_post=N("gpi"),
        sign=-1, weight_mean=0.8e-9, weight_std=0.1e-9,
        conn_prob=0.4, tau_syn=8e-3, plastic=False, dt=dt)

    # 7. STN -> GPi
    connections["stn_gpi"] = SynapseGroup(
        name="stn_gpi", N_pre=N("stn"), N_post=N("gpi"),
        sign=+1, weight_mean=1.5e-9, weight_std=0.25e-9,
        conn_prob=0.6, tau_syn=4e-3, plastic=False, dt=dt)

    # 8. GPi -> Thalamus (output gate)
    connections["gpi_thal"] = SynapseGroup(
        name="gpi_thal", N_pre=N("gpi"), N_post=N("thalamus"),
        sign=-1, weight_mean=2.0e-9, weight_std=0.3e-9,
        conn_prob=0.7, tau_syn=6e-3, plastic=False, dt=dt)

    # 9. Thalamus -> Motor Cortex
    connections["thal_ctx"] = SynapseGroup(
        name="thal_ctx", N_pre=N("thalamus"), N_post=N("cortex"),
        sign=+1, weight_mean=0.5e-9, weight_std=0.08e-9,
        conn_prob=0.3, tau_syn=4e-3, plastic=False, dt=dt)

    # 10. SNc -> Striatum (dopamine volume transmission)
    connections["snc_d1"] = SynapseGroup(
        name="snc_d1", N_pre=N("snc"), N_post=N("d1_msn"),
        sign=+1, weight_mean=0.4e-9, weight_std=0.06e-9,
        conn_prob=0.8, tau_syn=80e-3, plastic=False, dt=dt)

    connections["snc_d2"] = SynapseGroup(
        name="snc_d2", N_pre=N("snc"), N_post=N("d2_msn"),
        sign=-1, weight_mean=0.4e-9, weight_std=0.06e-9,
        conn_prob=0.8, tau_syn=80e-3, plastic=False, dt=dt)

    # 11. Serotonin -> GPi (risk/aversion)
    connections["sero_gpi"] = SynapseGroup(
        name="sero_gpi", N_pre=N("serotonin"), N_post=N("gpi"),
        sign=+1, weight_mean=0.3e-9, weight_std=0.05e-9,
        conn_prob=0.5, tau_syn=50e-3, plastic=False, dt=dt)

    # 12. Norepinephrine -> STN (arousal/uncertainty)
    connections["ne_stn"] = SynapseGroup(
        name="ne_stn", N_pre=N("norepinephrine"), N_post=N("stn"),
        sign=+1, weight_mean=0.3e-9, weight_std=0.05e-9,
        conn_prob=0.5, tau_syn=60e-3, plastic=False, dt=dt)

    # 13. Bayesian layer -> Striatum
    connections["bayes_striatum"] = SynapseGroup(
        name="bayes_striatum", N_pre=N("bayesian_layer"),
        N_post=N("d1_msn"),
        sign=+1, weight_mean=0.4e-9, weight_std=0.06e-9,
        conn_prob=0.4, tau_syn=10e-3, plastic=False, dt=dt)

    # 14. Bayesian layer -> GPi
    connections["bayes_gpi"] = SynapseGroup(
        name="bayes_gpi", N_pre=N("bayesian_layer"), N_post=N("gpi"),
        sign=+1, weight_mean=0.3e-9, weight_std=0.05e-9,
        conn_prob=0.3, tau_syn=10e-3, plastic=False, dt=dt)

    print(f"  [CONNECTIVITY] {len(connections)} synaptic groups built.")
    return connections