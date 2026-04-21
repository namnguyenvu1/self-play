"""
MBPO (Model-Based Policy Optimization) Agent — Dyna-type PPO implementation.

Continuous action changes vs discrete baseline
-----------------------------------------------
- train_dynamics_models():
    Discrete  → one-hot encode actions before feeding to dynamics model
    Continuous→ use tanh-squashed actions directly (no one-hot)

- generate_imagined_rollouts():
    Discrete  → sample Categorical, one-hot encode for dynamics
    Continuous→ sample Normal, apply tanh+rescale, use squashed action
                for dynamics; log-prob uses full SAC-style correction
"""

import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

from config import Config
from base_agent import BaseAgent
from networks import DynamicsModel
from buffers import RolloutBuffer, ReplayBuffer


class MBPOAgent(BaseAgent):
    def __init__(self, agent_id: str, obs_dim: int, act_dim: int, cfg: Config):
        super().__init__(agent_id, obs_dim, act_dim, cfg)

        self.dynamics_models = [
            DynamicsModel(
                obs_dim, act_dim, cfg.hidden_size,
                use_normalization=cfg.use_obs_normalization,
                normalization_momentum=cfg.normalization_momentum,
            ).to(self.device) for _ in range(cfg.num_models)
        ]

        self.dynamics_optimizers = [
            optim.Adam(model.parameters(), lr=cfg.lr_model)
            for model in self.dynamics_models
        ]
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
        self.dynamics_train_count = 0

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def add_to_replay_buffer(
        self,
        obs: np.ndarray,
        action,                  # int (discrete) or np.ndarray (continuous)
        next_obs: np.ndarray,
        reward: float,
    ):
        self.replay_buffer.add(obs, action, next_obs, reward)

    # ------------------------------------------------------------------
    # Action encoding helpers for dynamics model
    # ------------------------------------------------------------------

    def _encode_actions_for_dynamics(
        self,
        actions,                       # int64 tensor [B] or float32 tensor [B, act_dim]
    ) -> torch.Tensor:
        """
        Encode actions for input to the dynamics model.

        Discrete  : one-hot encode integer actions → [B, act_dim]
        Continuous: actions are already tanh-squashed float32 → pass through as-is
        """
        if self.cfg.is_continuous:
            # actions already float32 [B, act_dim] — pass directly
            return actions.float()
        else:
            return torch.nn.functional.one_hot(actions.long(), self.act_dim).float()

    # ------------------------------------------------------------------
    # Dynamics model training
    # ------------------------------------------------------------------

    def train_dynamics_models(self) -> Dict[str, float]:
        cfg = self.cfg
        if not self.replay_buffer.is_ready(cfg.model_batch_size):
            return {}

        all_dynamics_losses, all_reward_losses = [], []

        for _ in range(cfg.model_epochs):
            batch = self.replay_buffer.sample(cfg.model_batch_size)
            obs_list, act_list, next_obs_list, rew_list = zip(*batch)

            obs_t      = torch.tensor(np.array(obs_list),      dtype=torch.float32, device=self.device)
            next_obs_t = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=self.device)
            rew_t      = torch.tensor(np.array(rew_list),      dtype=torch.float32, device=self.device).reshape(-1, 1)

            if cfg.is_continuous:
                # act_list contains float32 arrays [act_dim]
                act_t = torch.tensor(np.array(act_list), dtype=torch.float32, device=self.device)
                # tanh-squashed actions already stored in replay buffer → use directly
                act_encoded = act_t                                            # [B, act_dim]
            else:
                act_t = torch.tensor(np.array(act_list), dtype=torch.int64, device=self.device)
                act_encoded = torch.nn.functional.one_hot(act_t, self.act_dim).float()  # [B, act_dim]

            obs_delta_target = next_obs_t - obs_t

            for model, optimizer in zip(self.dynamics_models, self.dynamics_optimizers):
                model.update_normalization_stats(obs_t)
                obs_delta_pred, reward_pred = model(obs_t, act_encoded)

                loss_dynamics = ((obs_delta_pred - obs_delta_target) ** 2).mean()
                loss_reward   = ((reward_pred - rew_t) ** 2).mean()
                loss = loss_dynamics + loss_reward

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_dynamics_losses.append(loss_dynamics.item())
                all_reward_losses.append(loss_reward.item())

        self.dynamics_train_count += 1
        return {
            "dynamics_loss":     np.mean(all_dynamics_losses),
            "reward_loss":       np.mean(all_reward_losses),
            "replay_buffer_size": len(self.replay_buffer),
        }

    # ------------------------------------------------------------------
    # Elite model selection
    # ------------------------------------------------------------------

    def _select_elite_models(self) -> List[int]:
        cfg = self.cfg
        if not self.replay_buffer.is_ready(500):
            return list(range(cfg.num_elites))

        val_size = min(len(self.replay_buffer) // 10, 2000)
        val_samples = random.sample(self.replay_buffer.get_all(), val_size)
        obs_list, act_list, next_obs_list, rew_list = zip(*val_samples)

        obs_t      = torch.tensor(np.array(obs_list),      dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=self.device)
        rew_t      = torch.tensor(np.array(rew_list),      dtype=torch.float32, device=self.device).reshape(-1, 1)

        if cfg.is_continuous:
            act_t = torch.tensor(np.array(act_list), dtype=torch.float32, device=self.device)
            act_encoded = act_t
        else:
            act_t = torch.tensor(np.array(act_list), dtype=torch.int64, device=self.device)
            act_encoded = torch.nn.functional.one_hot(act_t, self.act_dim).float()

        obs_delta_target = next_obs_t - obs_t

        losses = []
        for model in self.dynamics_models:
            with torch.no_grad():
                obs_delta_pred, reward_pred = model(obs_t, act_encoded)
                total_loss = (
                    ((obs_delta_pred - obs_delta_target) ** 2).mean()
                    + ((reward_pred - rew_t) ** 2).mean()
                )
                losses.append(total_loss.item())

        return np.argsort(losses)[:cfg.num_elites].tolist()

    # ------------------------------------------------------------------
    # Rollout length schedule
    # ------------------------------------------------------------------

    def _get_rollout_length(self) -> int:
        cfg = self.cfg
        progress = min(1.0, self.update_count / cfg.rollout_schedule_updates)
        return int(
            cfg.min_model_rollout_length
            + (cfg.max_model_rollout_length - cfg.min_model_rollout_length) * progress
        )

    # ------------------------------------------------------------------
    # Imagined rollout generation
    # ------------------------------------------------------------------

    def generate_imagined_rollouts(self, rollout_length: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate imagined rollouts using the learned dynamics ensemble.

        Discrete mode:
            - Sample Categorical; one-hot encode for dynamics model
        Continuous mode:
            - Sample Normal; apply tanh + rescale to [0,1]
            - Feed tanh-squashed action to dynamics model
            - Log-prob uses full SAC-style correction (matches select_action)

        Returns None if the replay buffer is not ready yet.
        """
        cfg = self.cfg
        imagination_batch_size = cfg.minibatch_size if cfg.minibatch_size > 0 else 256

        if not self.replay_buffer.is_ready(imagination_batch_size):
            return None

        start_samples = self.replay_buffer.sample(imagination_batch_size)
        current_obs_t = torch.tensor(
            np.array([s[0] for s in start_samples]),
            dtype=torch.float32, device=self.device,
        )

        elite_indices = self._select_elite_models()
        elite_models  = [self.dynamics_models[i] for i in elite_indices]

        obs_hist, act_hist, logp_hist, rew_hist, val_hist = [], [], [], [], []

        for _ in range(rollout_length):
            with torch.no_grad():
                if cfg.is_continuous:
                    # ---- Continuous action selection --------------------------
                    mean, log_std, values = self.model(current_obs_t)
                    std  = log_std.exp()
                    dist = Normal(mean, std)
                    raw_actions = dist.rsample()                          # pre-tanh u

                    # Squash + rescale to [0, 1]
                    squashed   = torch.tanh(raw_actions)
                    actions_01 = (squashed + 1.0) / 2.0                  # [B, act_dim]

                    # Corrected log-prob (SAC-style)
                    log_probs, _ = self._compute_log_prob_and_entropy(dist, raw_actions)  # [B]

                    # Dynamics model receives tanh-squashed action in [0,1]
                    act_encoded = actions_01                              # [B, act_dim]

                    # Store squashed action (consistent with replay buffer convention)
                    actions_to_store = actions_01.cpu().numpy()          # [B, act_dim]

                else:
                    # ---- Discrete action selection ----------------------------
                    logits, values = self.model(current_obs_t)
                    dist    = Categorical(logits=logits)
                    actions = dist.sample()                               # [B]
                    log_probs = dist.log_prob(actions)                    # [B]

                    act_encoded = torch.nn.functional.one_hot(actions, self.act_dim).float()
                    actions_to_store = actions.cpu().numpy()              # [B]

                # ---- Dynamics rollout (shared) --------------------------------
                all_obs_deltas, all_rewards = [], []
                for model in elite_models:
                    obs_delta, reward = model(current_obs_t, act_encoded)
                    all_obs_deltas.append(obs_delta)
                    all_rewards.append(reward)

                all_obs_deltas = torch.stack(all_obs_deltas)             # [M, B, obs_dim]
                all_rewards    = torch.stack(all_rewards)                # [M, B, 1]

                mean_obs_delta = all_obs_deltas.mean(dim=0)
                mean_reward    = all_rewards.mean(dim=0).squeeze(-1)     # [B]
                obs_variance   = all_obs_deltas.var(dim=0).mean(dim=-1)  # [B]

                penalty_max      = getattr(cfg, 'uncertainty_penalty_max', 1.0)
                raw_penalty      = cfg.uncertainty_penalty_coef * obs_variance
                clipped_penalty  = torch.clamp(raw_penalty, 0.0, penalty_max)
                penalized_reward = mean_reward - clipped_penalty

                next_obs_t = current_obs_t + mean_obs_delta

            unnorm_values = self.denormalize_value(values.cpu().numpy())

            obs_hist.append(current_obs_t.cpu().numpy())
            act_hist.append(actions_to_store)
            logp_hist.append(log_probs.cpu().numpy())
            rew_hist.append(penalized_reward.cpu().numpy())
            val_hist.append(unnorm_values)

            current_obs_t = next_obs_t

        # Bootstrap value for last state
        with torch.no_grad():
            if cfg.is_continuous:
                _, _, last_values = self.model(current_obs_t)
            else:
                _, last_values = self.model(current_obs_t)
            last_values = self.denormalize_value(last_values.cpu().numpy())

        # Stack arrays: [Time, Batch, ...]
        obs_arr  = np.stack(obs_hist)    # [T, B, obs_dim]
        act_arr  = np.stack(act_hist)    # [T, B] or [T, B, act_dim]
        logp_arr = np.stack(logp_hist)   # [T, B]
        rew_arr  = np.stack(rew_hist)    # [T, B]
        val_arr  = np.stack(val_hist)    # [T, B]

        # GAE computation (batched, no episode boundaries in imagination)
        adv_arr = np.zeros_like(rew_arr)
        ret_arr = np.zeros_like(rew_arr)
        gae     = np.zeros(imagination_batch_size)

        gamma, lam = cfg.gamma, cfg.lam
        for t in reversed(range(rollout_length)):
            next_val = last_values if t == rollout_length - 1 else val_arr[t + 1]
            delta    = rew_arr[t] + gamma * next_val - val_arr[t]
            gae      = delta + gamma * lam * gae
            adv_arr[t] = gae
            ret_arr[t] = gae + val_arr[t]

        # Flatten time × batch into a single batch dimension
        B = imagination_batch_size
        if cfg.is_continuous:
            actions_flat = act_arr.reshape(-1, self.act_dim)             # [T*B, act_dim]
        else:
            actions_flat = act_arr.reshape(-1)                           # [T*B]

        return {
            'obs':        obs_arr.reshape(-1, self.obs_dim),
            'actions':    actions_flat,
            'old_logp':   logp_arr.reshape(-1),
            'returns':    ret_arr.reshape(-1),
            'advantages': adv_arr.reshape(-1),
            'old_values': val_arr.reshape(-1),
        }

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, last_value: float) -> Dict[str, float]:
        metrics = {}

        # 1. Train dynamics models on real data
        metrics.update(self.train_dynamics_models())

        # 2. Train PPO on real rollout data
        real_metrics = self.update_policy(self.buffer, last_value, is_imagined=False)
        if "policy_loss" in real_metrics:
            metrics["real_policy_loss"] = real_metrics["policy_loss"]
        metrics.update(real_metrics)

        # 3. Generate imagined rollouts
        rollout_length = self._get_rollout_length()
        imagined_data  = self.generate_imagined_rollouts(rollout_length)

        # 4. Train PPO on imagined data
        if imagined_data is not None:
            imag_metrics = self._run_ppo_epochs(**imagined_data)
            metrics.update(imag_metrics)
            n_imagined = len(imagined_data['obs'])
        else:
            n_imagined = 0

        metrics["rollout_length"]       = rollout_length
        metrics["imagined_transitions"] = n_imagined

        self.update_count += 1
        return metrics