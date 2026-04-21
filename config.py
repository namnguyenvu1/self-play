"""
Configuration module for Multi-Agent PPO/MBPO training.
"""

from dataclasses import dataclass, field
from typing import Literal
import torch

@dataclass
class Config:
    # ==================== Environment ====================
    env_name: str = "simple_adversary_v3"

    # ==================== Action Space ====================
    # Toggle between discrete and continuous action spaces.
    # When "continuous", MPE is created with continuous_actions=True
    # and the policy uses a Diagonal Gaussian + tanh squashing.
    action_space_type: Literal["discrete", "continuous"] = "discrete"

    # ==================== Agent Assignment ====================
    good_agent_algo: Literal["ppo", "mbpo"] = "mbpo"
    adversary_algo: Literal["ppo", "mbpo"] = "ppo"

    # ==================== Network Architecture ====================
    hidden_size: int = 128

    # ==================== PPO Hyperparameters ====================
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3  # Often slightly higher in PPO papers
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2

    # Paper Recommendation: 10 epochs on difficult envs, 15 on easy.
    epochs: int = 10

    # Paper Recommendation: Avoid splitting into mini-batches.
    # Set to 0 or equal to rollout_steps to use Full Batch.
    minibatch_size: int = 0

    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # PPO Specific Features (Paper Compliance)
    use_value_normalization: bool = True
    use_obs_normalization: bool = True
    use_clipped_value_loss: bool = True

    # ==================== Continuous Action Space Hyperparameters ====================
    # Only used when action_space_type == "continuous"

    # Initial value for the learned log_std parameter
    log_std_init: float = -0.5

    # Clamp bounds for log_std to keep std in a stable range
    # std range: [exp(-3) ≈ 0.05,  exp(1) ≈ 2.72]
    log_std_min: float = -3.0
    log_std_max: float = 1.0

    # ==================== MBPO Hyperparameters ====================
    lr_model: float = 1e-3
    model_epochs: int = 20
    model_batch_size: int = 256
    replay_buffer_size: int = 1_000_000
    num_models: int = 7
    num_elites: int = 5
    min_model_rollout_length: int = 1
    max_model_rollout_length: int = 10
    rollout_schedule_updates: int = 250
    uncertainty_penalty_coef: float = 0.5
    uncertainty_penalty_max: float = 1.0
    normalization_momentum: float = 0.99

    # ==================== Training Settings ====================
    # Paper uses large batch sizes (e.g. 60k steps in StarCraft, but 2048 is fine for MPE)
    rollout_steps: int = 2048
    max_updates: int = 250

    # ==================== Device ====================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_interval: int = 10

    def __post_init__(self):
        assert self.good_agent_algo in ["ppo", "mbpo"]
        assert self.adversary_algo in ["ppo", "mbpo"]
        assert self.action_space_type in ["discrete", "continuous"]
        assert self.log_std_min < self.log_std_max, "log_std_min must be less than log_std_max"
        assert self.log_std_min <= self.log_std_init <= self.log_std_max, (
            f"log_std_init ({self.log_std_init}) must be within "
            f"[log_std_min={self.log_std_min}, log_std_max={self.log_std_max}]"
        )

    @property
    def is_continuous(self) -> bool:
        """Convenience property to check if continuous action space is active."""
        return self.action_space_type == "continuous"