"""
Neural network architectures for multi-agent RL.

This module provides:
- ActorCritic: Shared network for policy and value function (PPO)
  - Discrete mode: Categorical distribution (unchanged)
  - Continuous mode: Diagonal Gaussian with learned log_std parameter
- DynamicsModel: Ensemble model for environment dynamics (MBPO)

All networks include proper initialization and normalization.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Supports both discrete (Categorical) and continuous (Diagonal Gaussian)
    action spaces, toggled via the `is_continuous` flag.

    Discrete mode:
        forward() → (logits [B, act_dim], value [B])

    Continuous mode:
        forward() → (mean [B, act_dim], log_std_clamped [act_dim], value [B])
        - mean  : raw pre-tanh mean output from the policy head
        - log_std: a learned nn.Parameter (state-independent), clamped to
                   [log_std_min, log_std_max] before returning
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        use_obs_normalization: bool = False,
        is_continuous: bool = False,
        log_std_init: float = -0.5,
        log_std_min: float = -3.0,
        log_std_max: float = 1.0,
    ):
        """
        Args:
            obs_dim:              Observation space dimension.
            act_dim:              Number of discrete actions  OR
                                  continuous action dimension.
            hidden_size:          Hidden layer width.
            use_obs_normalization: Whether to apply running-stat obs normalisation.
            is_continuous:        If True, use Diagonal Gaussian policy head.
            log_std_init:         Initial value for the learned log_std parameter.
            log_std_min:          Lower clamp bound for log_std.
            log_std_max:          Upper clamp bound for log_std.
        """
        super().__init__()

        self.is_continuous = is_continuous
        self.use_obs_normalization = use_obs_normalization
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ------------------------------------------------------------------
        # Running statistics for observation normalization
        # ------------------------------------------------------------------
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_var', torch.ones(obs_dim))
        self.register_buffer('obs_count', torch.zeros(1))

        # ------------------------------------------------------------------
        # Shared feature extractor
        # ------------------------------------------------------------------
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Policy head (actor)
        #   Discrete  → logits over actions
        #   Continuous → mean of Gaussian (pre-tanh)
        # ------------------------------------------------------------------
        self.policy = nn.Linear(hidden_size, act_dim)

        # ------------------------------------------------------------------
        # Continuous-only: learned state-independent log_std
        #   Shape [act_dim] so each action dimension has its own std.
        #   Stored as nn.Parameter so it is updated by the optimizer.
        # ------------------------------------------------------------------
        if self.is_continuous:
            self.log_std = nn.Parameter(
                torch.full((act_dim,), log_std_init)
            )
        else:
            self.log_std = None  # not used in discrete mode

        # ------------------------------------------------------------------
        # Value head (critic) — shared across both modes
        # ------------------------------------------------------------------
        self.value = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self):
        """Orthogonal initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    # Observation normalisation helpers
    # ------------------------------------------------------------------

    def update_obs_stats(self, obs: torch.Tensor):
        """Update running mean and variance for observation normalization."""
        if not self.use_obs_normalization:
            return
        with torch.no_grad():
            batch_mean = obs.mean(dim=0)
            batch_var = obs.var(dim=0, unbiased=False)
            batch_count = obs.shape[0]

            if self.obs_count.item() == 0:
                self.obs_mean.copy_(batch_mean)
                self.obs_var.copy_(batch_var)
                self.obs_count += batch_count
            else:
                delta = batch_mean - self.obs_mean
                tot_count = self.obs_count + batch_count

                new_mean = self.obs_mean + delta * batch_count / tot_count
                m_a = self.obs_var * self.obs_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + (delta ** 2) * self.obs_count * batch_count / tot_count
                new_var = M2 / tot_count

                self.obs_mean.copy_(new_mean)
                self.obs_var.copy_(new_var)
                self.obs_count.copy_(tot_count)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply running-stat normalisation to observations."""
        if not self.use_obs_normalization or self.obs_count.item() == 0:
            return obs
        norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return torch.clamp(norm_obs, -10.0, 10.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Returns:
            Discrete mode  : (logits [B, act_dim],  value [B])
            Continuous mode: (mean   [B, act_dim],
                              log_std_clamped [act_dim],
                              value  [B])
        """
        x_norm = self.normalize_obs(x)
        features = self.net(x_norm)
        value = self.value(features).squeeze(-1)

        if self.is_continuous:
            mean = self.policy(features)
            # Clamp log_std to keep std in a numerically stable range
            log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            return mean, log_std_clamped, value
        else:
            logits = self.policy(features)
            return logits, value


class DynamicsModel(nn.Module):
    """
    Dynamics model for MBPO (predicts next state and reward).

    Key Features:
    - Predicts state delta (next_obs - obs) instead of absolute next_obs
    - Separate prediction heads for dynamics and reward (CRITICAL FIX)
    - Built-in observation normalization with running statistics

    Input:
        Discrete mode  : [obs, action_one_hot]   → input dim = obs_dim + act_dim
        Continuous mode: [obs, action_continuous] → input dim = obs_dim + act_dim
        (act_dim means different things in each mode but the network
         structure is identical — the difference is upstream in the agent.)

    Output: [obs_delta, reward]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        use_normalization: bool = True,
        normalization_momentum: float = 0.99
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_normalization = use_normalization
        self.momentum = normalization_momentum

        # Running statistics for observation normalization
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))
        self.register_buffer('normalization_count', torch.zeros(1))

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # CRITICAL FIX: Separate heads for dynamics and reward
        self.dynamics_head = nn.Linear(hidden_size, obs_dim)  # Predicts obs_delta
        self.reward_head = nn.Linear(hidden_size, 1)          # Predicts reward

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def update_normalization_stats(self, obs_batch: torch.Tensor):
        if not self.use_normalization:
            return
        with torch.no_grad():
            batch_mean = obs_batch.mean(dim=0)
            batch_std = obs_batch.std(dim=0) + 1e-8

            if self.normalization_count.item() == 0:
                self.obs_mean.copy_(batch_mean)
                self.obs_std.copy_(batch_std)
            else:
                self.obs_mean = self.momentum * self.obs_mean + (1 - self.momentum) * batch_mean
                self.obs_std = self.momentum * self.obs_std + (1 - self.momentum) * batch_std

            self.normalization_count += 1

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.use_normalization or self.normalization_count.item() == 0:
            return obs
        norm_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return torch.clamp(norm_obs, -10.0, 10.0)

    def forward(
        self,
        obs: torch.Tensor,
        act_encoded: torch.Tensor          # one-hot (discrete) OR raw tanh action (continuous)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:         [B, obs_dim]
            act_encoded: [B, act_dim]  — one-hot for discrete, tanh-squashed for continuous
        Returns:
            obs_delta: [B, obs_dim]
            reward:    [B, 1]
        """
        obs_normalized = self.normalize_obs(obs)
        x = torch.cat([obs_normalized, act_encoded], dim=-1)
        features = self.shared_net(x)
        obs_delta = self.dynamics_head(features)
        reward = self.reward_head(features)
        return obs_delta, reward

    def predict_next_state(
        self,
        obs: torch.Tensor,
        act_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_delta, reward = self.forward(obs, act_encoded)
        next_obs = obs + obs_delta
        return next_obs, reward


class DynamicsEnsemble:
    """
    Ensemble of dynamics models for uncertainty estimation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_models: int,
        hidden_size: int = 128,
        device: str = "cpu",
        use_normalization: bool = True
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_models = num_models
        self.device = device

        self.models = [
            DynamicsModel(
                obs_dim,
                act_dim,
                hidden_size,
                use_normalization=use_normalization
            ).to(device)
            for _ in range(num_models)
        ]

    def __getitem__(self, idx: int) -> DynamicsModel:
        return self.models[idx]

    def __len__(self) -> int:
        return self.num_models

    def predict_with_uncertainty(
        self,
        obs: torch.Tensor,
        act_encoded: torch.Tensor,
        elite_indices: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state using ensemble and compute uncertainty.

        Args:
            obs:          [B, obs_dim]
            act_encoded:  [B, act_dim]  one-hot (discrete) or tanh action (continuous)
            elite_indices: indices of elite models (None = use all)

        Returns:
            next_obs_mean: [B, obs_dim]
            reward_mean:   [B, 1]
            obs_variance:  [B]
        """
        models_to_use = elite_indices if elite_indices else range(self.num_models)

        with torch.no_grad():
            predictions = []
            for idx in models_to_use:
                next_obs, reward = self.models[idx].predict_next_state(obs, act_encoded)
                predictions.append(torch.cat([next_obs, reward], dim=-1))

            predictions = torch.stack(predictions)       # [M, B, obs_dim+1]
            mean_pred = predictions.mean(dim=0)          # [B, obs_dim+1]
            next_obs_mean = mean_pred[:, :-1]
            reward_mean = mean_pred[:, -1:]
            obs_variance = predictions[:, :, :-1].var(dim=0).mean(dim=-1)  # [B]

        return next_obs_mean, reward_mean, obs_variance