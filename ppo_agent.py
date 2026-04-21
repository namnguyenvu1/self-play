"""
Pure PPO Agent implementation.

No changes required here — all discrete/continuous branching is handled
in BaseAgent. This class simply calls the inherited update_policy().
"""

from typing import Dict
from config import Config
from base_agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, agent_id: str, obs_dim: int, act_dim: int, cfg: Config):
        super().__init__(agent_id, obs_dim, act_dim, cfg)

    def update(self, last_value: float) -> Dict[str, float]:
        metrics = self.update_policy(self.buffer, last_value, is_imagined=False)
        self.update_count += 1
        return metrics