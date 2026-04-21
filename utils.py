"""
Utility functions for multi-agent training.
"""

import sys
import numpy as np
from typing import TextIO


class RunningMeanStd:
    """Tracks running mean and variance for normalization."""
    def __init__(self, shape=()):
        self.mean  = np.zeros(shape, 'float64')
        self.var   = np.ones(shape,  'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x,  axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a      = self.var * self.count
        m_b      = batch_var * batch_count
        M2       = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var  = M2 / tot_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot_count


def create_env(env_name: str, continuous_actions: bool = False):
    """
    Create a PettingZoo MPE environment.

    Args:
        env_name:          Short environment name (e.g. 'simple_adversary').
        continuous_actions: If True, create the environment with continuous
                            action spaces (Box).  MPE supports this flag on
                            all standard environments.
    """
    from pettingzoo.mpe import (
        simple_adversary_v3,
        simple_crypto_v3,
        simple_push_v3,
        simple_reference_v3,
        simple_speaker_listener_v3,
        simple_spread_v3,
        simple_tag_v3,
        simple_world_comm_v3,
    )

    env_map = {
        'simple_adversary':        simple_adversary_v3,
        'simple_crypto':           simple_crypto_v3,
        'simple_push':             simple_push_v3,
        'simple_reference':        simple_reference_v3,
        'simple_speaker_listener': simple_speaker_listener_v3,
        'simple_spread':           simple_spread_v3,
        'simple_tag':              simple_tag_v3,
        'simple_world_comm':       simple_world_comm_v3,
    }

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}")

    return env_map[env_name].parallel_env(continuous_actions=continuous_actions)


class DualOutput:
    def __init__(self, file_obj: TextIO, terminal: TextIO):
        self.file     = file_obj
        self.terminal = terminal

    def write(self, message: str):
        self.file.write(message)
        self.terminal.write(message)

    def flush(self):
        self.file.flush()
        self.terminal.flush()


def setup_logging(output_file: str = None):
    if output_file is None:
        return None, None
    original_stdout = sys.stdout
    log_file        = open(output_file, 'w')
    sys.stdout      = DualOutput(log_file, original_stdout)
    return original_stdout, log_file


def restore_logging(original_stdout, log_file):
    if original_stdout is not None:
        sys.stdout = original_stdout
    if log_file is not None:
        log_file.close()


def print_config_summary(cfg):
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Env: {cfg.env_name} | Device: {cfg.device}")
    print(f"Action Space: {cfg.action_space_type.upper()}")
    print(f"Good: {cfg.good_agent_algo.upper()} | Adv: {cfg.adversary_algo.upper()}")
    print(f"Updates: {cfg.max_updates} | Rollout: {cfg.rollout_steps}")
    print(f"Epochs: {cfg.epochs} | Minibatch: {cfg.minibatch_size} (0=Full Batch)")
    print(f"Value Norm: {cfg.use_value_normalization} | Obs Norm: {cfg.use_obs_normalization}")
    if cfg.is_continuous:
        print(f"log_std: init={cfg.log_std_init} | clamp=[{cfg.log_std_min}, {cfg.log_std_max}]")
    print("=" * 70 + "\n")


def format_time(seconds: float) -> str:
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def save_training_history(history: dict, filename: str):
    import pickle
    if filename.endswith('.npz'):
        np.savez_compressed(filename, **history)
    elif filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(history, f)