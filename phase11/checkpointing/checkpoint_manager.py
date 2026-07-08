"""
CheckpointManager — saves and restores full pipeline state.
Saves: actor preferences, critic weights, prior P(a),
       synaptic weights, neuromodulator omegas, metrics.
"""

import numpy as np
import json
import os
from datetime import datetime


class CheckpointManager:

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_count = 0

    def save(self,
              step         : int,
              episode      : int,
              actor        : object,
              multi_critic : object,
              pipeline_p3  : object,
              nm_ctrl      : object,
              plast        : object,
              metrics_summary: dict) -> str:
        """
        Saves a checkpoint. Returns path to saved file.
        """
        self.save_count += 1
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ckpt_ep{episode:03d}_step{step:07d}_{ts}.json"
        path = os.path.join(self.checkpoint_dir, name)

        ckpt = {
            "step"    : step,
            "episode" : episode,
            "timestamp": ts,

            # Actor state
            "actor_preference": actor.action_preference.tolist(),
            "actor_pi"         : actor.pi.tolist(),
            "actor_temperature": float(actor.temperature),

            # Prior P(a)
            "prior"           : pipeline_p3.encoder.prior_tracker.prior.tolist(),

            # Neuromodulator fusion weights
            "omega_d"         : float(nm_ctrl.fusion.omega_d),
            "omega_s"         : float(nm_ctrl.fusion.omega_s),
            "omega_n"         : float(nm_ctrl.fusion.omega_n),

            # Critic means (not full weights — too large for JSON)
            "critic_means"    : {
                c.name: float(np.mean(np.abs(c.W_v)))
                for c in multi_critic.critics
            },

            # Plasticity weight means per connection
            "weight_means"    : {
                name: float(np.mean(np.abs(eng.W)))
                for name, eng in plast.engines.items()
            },

            # Summary metrics
            "metrics"         : metrics_summary,
        }

        with open(path, "w") as f:
            json.dump(ckpt, f, indent=2)

        print(f"  [CKPT] Saved checkpoint: {name}")
        return path

    def list_checkpoints(self) -> list:
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith("ckpt_") and f.endswith(".json")]
        return sorted(files)

    def load_latest(self) -> dict:
        files = self.list_checkpoints()
        if not files:
            return {}
        path = os.path.join(self.checkpoint_dir, files[-1])
        with open(path) as f:
            return json.load(f)