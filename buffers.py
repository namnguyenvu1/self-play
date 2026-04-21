"""
Buffer classes for storing and managing training data.

This module provides:
- RolloutBuffer: For on-policy PPO rollouts (discrete and continuous actions)
- ReplayBuffer:  For off-policy MBPO dynamics training

Action storage is dtype-agnostic:
  Discrete   → stored and returned as np.int64 scalars
  Continuous → stored and returned as np.float32 vectors [act_dim]
The buffer detects which case applies from the first element added.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Union
import random


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data for PPO.

    Stores transitions (obs, action, log_prob, reward, done, value)
    and computes returns and advantages using GAE.

    Action dtype handling
    ----------------------
    - Discrete  : action is an int  → stored as int, returned as np.int64 array
    - Continuous: action is a float32 ndarray [act_dim] → returned as np.float32 [T, act_dim]
    The buffer stores whatever is passed; dtype is inferred at retrieval time.
    """

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[Union[int, np.ndarray]] = []
        self.log_prob: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        """
        Add a single transition to the buffer.

        Args:
            obs:      Observation array.
            action:   int (discrete) or np.ndarray float32 (continuous).
            log_prob: Log-probability of the action under the behaviour policy.
            reward:   Scalar reward.
            done:     Episode termination flag.
            value:    Estimated value (un-normalised) from the critic.
        """
        self.obs.append(np.asarray(obs, dtype=np.float32))

        # Store action generically — preserve type for correct downstream casting
        if isinstance(action, np.ndarray):
            self.actions.append(action.astype(np.float32))
        else:
            self.actions.append(int(action))

        self.log_prob.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self):
        """Clear all stored data."""
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)

    def _stack_actions(self) -> np.ndarray:
        """
        Stack actions into a single array with the correct dtype.

        Discrete  : returns int64  array of shape [T]
        Continuous: returns float32 array of shape [T, act_dim]
        """
        if len(self.actions) == 0:
            return np.array([], dtype=np.int64)

        if isinstance(self.actions[0], np.ndarray):
            # Continuous — each element is [act_dim]
            return np.stack(self.actions, axis=0).astype(np.float32)
        else:
            # Discrete — each element is a Python int
            return np.array(self.actions, dtype=np.int64)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Bootstrap value for the last state.
            gamma:      Discount factor.
            lam:        GAE lambda parameter.

        Returns:
            obs:        [T, obs_dim]   float32
            actions:    [T]            int64   (discrete)
                        [T, act_dim]   float32 (continuous)
            old_logp:   [T]            float32
            returns:    [T]            float32
            advantages: [T]            float32
            old_values: [T]            float32
        """
        returns = []
        advantages = []
        gae = 0.0

        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[step])
            delta = self.rewards[step] + gamma * values[step + 1] * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return (
            np.array(self.obs, dtype=np.float32),
            self._stack_actions(),
            np.array(self.log_prob, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(advantages, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )


class ReplayBuffer:
    """
    Circular replay buffer for off-policy data storage (MBPO).

    Stores transitions (obs, action, next_obs, reward) for training
    dynamics models.

    Action storage:
        Discrete  : action stored as int
        Continuous: action stored as float32 ndarray [act_dim]
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        next_obs: np.ndarray,
        reward: float,
    ):
        """
        Add a transition to the buffer.

        Args:
            obs:      Current observation.
            action:   int (discrete) or float32 ndarray (continuous).
            next_obs: Next observation.
            reward:   Scalar reward.
        """
        if isinstance(action, np.ndarray):
            stored_action = action.astype(np.float32)
        else:
            stored_action = int(action)

        self.buffer.append((
            np.asarray(obs, dtype=np.float32),
            stored_action,
            np.asarray(next_obs, dtype=np.float32),
            float(reward),
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_all(self):
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size