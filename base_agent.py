"""
Base agent class with shared PPO logic.

Supports both discrete (Categorical) and continuous (Diagonal Gaussian + tanh)
action spaces, controlled by cfg.action_space_type.

Continuous action pipeline
--------------------------
1. policy head outputs raw mean  u  (pre-tanh)
2. tanh squash:        tanh_u = tanh(u)         ∈ (-1, 1)
3. rescale to [0, 1]:  a      = (tanh_u + 1) / 2
4. log-prob (SAC-style correction):
       log π(a) = Σ [ log N(u_i; μ_i, σ_i)
                       - log(1 - tanh²(u_i))
                       - log(0.5) ]          ← jacobian of (·+1)/2
5. Entropy: pre-squash Gaussian entropy (approximation, standard for PPO)
"""

from abc import ABC, abstractmethod
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Dict, Any, Union

from config import Config
from networks import ActorCritic
from buffers import RolloutBuffer
from utils import RunningMeanStd


class BaseAgent(ABC):
    def __init__(self, agent_id: str, obs_dim: int, act_dim: int, cfg: Config):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Build ActorCritic with continuous flag and log_std hyperparameters
        self.model = ActorCritic(
            obs_dim,
            act_dim,
            cfg.hidden_size,
            use_obs_normalization=cfg.use_obs_normalization,
            is_continuous=cfg.is_continuous,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(self.device)

        self.ret_rms = RunningMeanStd(shape=())

        # Include log_std in the actor parameter group when continuous
        if cfg.is_continuous:
            self.optimizer = optim.Adam([
                {"params": self.model.net.parameters(),    "lr": cfg.lr_actor},
                {"params": self.model.policy.parameters(), "lr": cfg.lr_actor},
                {"params": [self.model.log_std],           "lr": cfg.lr_actor},
                {"params": self.model.value.parameters(),  "lr": cfg.lr_critic},
            ])
        else:
            self.optimizer = optim.Adam([
                {"params": self.model.net.parameters(),    "lr": cfg.lr_actor},
                {"params": self.model.policy.parameters(), "lr": cfg.lr_actor},
                {"params": self.model.value.parameters(),  "lr": cfg.lr_critic},
            ])

        self.buffer = RolloutBuffer()
        self.update_count = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Value normalisation helpers
    # ------------------------------------------------------------------

    def denormalize_value(self, norm_value):
        if not self.cfg.use_value_normalization:
            return norm_value
        return norm_value * np.sqrt(self.ret_rms.var + 1e-8) + self.ret_rms.mean

    # ------------------------------------------------------------------
    # Log-prob and entropy helpers (continuous)
    # ------------------------------------------------------------------

    @staticmethod
    def _tanh_squash_correction(raw_action: torch.Tensor) -> torch.Tensor:
        """
        SAC-style log-prob correction for tanh squashing + [0,1] rescaling.

        Full jacobian:
            a = (tanh(u) + 1) / 2
            da/du = (1 - tanh²(u)) / 2
            log|da/du| = log(1 - tanh²(u)) - log(2)

        Summed over action dimensions → scalar per sample.

        Args:
            raw_action: pre-tanh values u  [B, act_dim]
        Returns:
            correction: [B]  (to be *subtracted* from Gaussian log-prob)
        """
        # log(1 - tanh²(u))  — numerically stable version
        log_one_minus_tanh_sq = 2.0 * (
            np.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        # log(2) term from the /2 rescaling
        log_half = np.log(0.5)
        correction = (log_one_minus_tanh_sq + log_half).sum(dim=-1)
        return correction

    def _compute_log_prob_and_entropy(
        self,
        dist: Normal,
        raw_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected log-prob and (approximate) entropy for continuous policy.

        Args:
            dist:       Normal distribution over pre-tanh actions.
            raw_action: Pre-tanh action u  [B, act_dim].
        Returns:
            log_prob: [B]   corrected log-probability of the squashed action
            entropy:  [B]   pre-squash Gaussian entropy (approximation)
        """
        # Gaussian log-prob summed over action dimensions
        log_prob_gaussian = dist.log_prob(raw_action).sum(dim=-1)  # [B]

        # Subtract tanh + rescaling jacobian
        correction = self._tanh_squash_correction(raw_action)       # [B]
        log_prob = log_prob_gaussian - correction                    # [B]

        # Pre-squash entropy (approximation standard for PPO)
        entropy = dist.entropy().sum(dim=-1)                         # [B]

        return log_prob, entropy

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Union[int, np.ndarray], float, float]:
        """
        Select an action given an observation.

        Returns:
            action:   int (discrete) or np.ndarray float32 [act_dim] (continuous)
            log_prob: float
            value:    float  (un-normalised)
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.cfg.is_continuous:
                mean, log_std, norm_value = self.model(obs_tensor)
                std = log_std.exp()
                dist = Normal(mean, std)

                if deterministic:
                    raw_action = mean
                else:
                    raw_action = dist.rsample()

                # Squash and rescale to [0, 1]
                squashed = torch.tanh(raw_action)
                action_t = (squashed + 1.0) / 2.0                       # [1, act_dim]

                log_prob, _ = self._compute_log_prob_and_entropy(dist, raw_action)

                action_out = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            else:
                logits, norm_value = self.model(obs_tensor)
                dist = Categorical(logits=logits)

                if deterministic:
                    action_t = logits.argmax(dim=-1)
                else:
                    action_t = dist.sample()

                log_prob = dist.log_prob(action_t)
                action_out = int(action_t.item())

            unnorm_value = self.denormalize_value(norm_value.item())

        return action_out, float(log_prob.item()), float(unnorm_value)

    # ------------------------------------------------------------------
    # Policy update (called from update_policy)
    # ------------------------------------------------------------------

    def update_policy(
        self,
        buffer: RolloutBuffer,
        last_value: float,
        is_imagined: bool = False,
    ) -> Dict[str, float]:
        if len(buffer) == 0:
            return {}

        obs, actions, old_logp, returns, advantages, old_values = (
            buffer.compute_returns_and_advantages(last_value, self.cfg.gamma, self.cfg.lam)
        )

        # Update normalizers ONLY on real data to prevent contamination
        if self.cfg.use_obs_normalization and not is_imagined:
            self.model.update_obs_stats(
                torch.tensor(obs, dtype=torch.float32, device=self.device)
            )

        if self.cfg.use_value_normalization and not is_imagined:
            self.ret_rms.update(returns)

        metrics = self._run_ppo_epochs(obs, actions, old_logp, returns, advantages, old_values)
        buffer.clear()
        return metrics

    # ------------------------------------------------------------------
    # PPO epoch loop
    # ------------------------------------------------------------------

    def _run_ppo_epochs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,          # int64 [T] or float32 [T, act_dim]
        old_logp: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        old_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run PPO update epochs over the provided data.

        Works for both discrete and continuous action spaces.
        """
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t        = torch.tensor(obs,        dtype=torch.float32, device=self.device)
        old_logp_t   = torch.tensor(old_logp,   dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Actions tensor — dtype depends on action space
        if self.cfg.is_continuous:
            actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
            # actions_t shape: [T, act_dim]
        else:
            actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)

        # Value normalisation
        if self.cfg.use_value_normalization:
            returns_normalized    = (returns    - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8)
            old_values_normalized = (old_values - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8)
        else:
            returns_normalized    = returns
            old_values_normalized = old_values

        returns_t    = torch.tensor(returns_normalized,    dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(old_values_normalized, dtype=torch.float32, device=self.device)

        policy_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

        for _ in range(self.cfg.epochs):
            indices = np.random.permutation(len(obs_t))
            mb_size = self.cfg.minibatch_size if self.cfg.minibatch_size > 0 else len(obs_t)

            for start in range(0, len(obs_t), mb_size):
                mb_idx = indices[start: start + mb_size]

                if self.cfg.is_continuous:
                    # ---- Continuous branch ----------------------------------------
                    mean, log_std, values = self.model(obs_t[mb_idx])
                    std  = log_std.exp()
                    dist = Normal(mean, std)

                    # Recover the pre-tanh raw action from the stored squashed action:
                    #   stored action a ∈ [0,1]  →  tanh_val = 2a - 1  →  u = arctanh(tanh_val)
                    squashed_stored = actions_t[mb_idx] * 2.0 - 1.0          # back to (-1,1)
                    # Clamp for numerical safety before atanh
                    squashed_clamped = squashed_stored.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    raw_action = torch.atanh(squashed_clamped)                # pre-tanh u

                    new_logp, entropy = self._compute_log_prob_and_entropy(dist, raw_action)

                else:
                    # ---- Discrete branch ------------------------------------------
                    logits, values = self.model(obs_t[mb_idx])
                    dist    = Categorical(logits=logits)
                    new_logp = dist.log_prob(actions_t[mb_idx])
                    entropy  = dist.entropy()

                # ---- Shared PPO objective -----------------------------------------
                ratio  = torch.exp(new_logp - old_logp_t[mb_idx])
                surr1  = ratio * advantages_t[mb_idx]
                surr2  = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advantages_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.cfg.use_clipped_value_loss:
                    value_losses_unclipped = (values - returns_t[mb_idx]) ** 2
                    value_pred_clipped = old_values_t[mb_idx] + torch.clamp(
                        values - old_values_t[mb_idx], -self.cfg.clip_eps, self.cfg.clip_eps
                    )
                    value_losses_clipped = (value_pred_clipped - returns_t[mb_idx]) ** 2
                    value_loss = torch.max(value_losses_unclipped, value_losses_clipped).mean()
                else:
                    value_loss = ((returns_t[mb_idx] - values) ** 2).mean()

                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef  * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean()
                    clip_fractions.append(clip_fraction.item())

        return {
            "policy_loss":   np.mean(policy_losses),
            "value_loss":    np.mean(value_losses),
            "entropy_loss":  np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
        }

    # ------------------------------------------------------------------
    # Abstract / shared interface
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self, last_value: float) -> Dict[str, float]:
        pass

    def add_experience(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.buffer.add(obs, action, log_prob, reward, done, value)
        self.total_steps += 1

    def get_buffer_size(self) -> int:
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        checkpoint = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count':         self.update_count,
            'total_steps':          self.total_steps,
            'ret_rms_mean':         self.ret_rms.mean,
            'ret_rms_var':          self.ret_rms.var,
            'ret_rms_count':        self.ret_rms.count,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count  = checkpoint['update_count']
        self.total_steps   = checkpoint['total_steps']
        self.ret_rms.mean  = checkpoint.get('ret_rms_mean',  np.zeros(()))
        self.ret_rms.var   = checkpoint.get('ret_rms_var',   np.ones(()))
        self.ret_rms.count = checkpoint.get('ret_rms_count', 1e-4)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'agent_id':    self.agent_id,
            'update_count': self.update_count,
            'total_steps':  self.total_steps,
            'buffer_size':  len(self.buffer),
        }