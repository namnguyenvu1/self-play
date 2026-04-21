# tdmpc2/envs/mpe.py
from pettingzoo.mpe import (
    simple_spread_v3, simple_speaker_listener_v4, simple_reference_v3, simple_adversary_v3,
    simple_crypto_v3, simple_push_v3, simple_tag_v3, simple_world_comm_v3
)
from pettingzoo.utils import aec_to_parallel
import gymnasium as gym
import numpy as np

# A dictionary to map task names to their respective environment creation functions
MPE_ENV_MAP = {
    "simple_spread": simple_spread_v3.env,
    "simple_speaker_listener": simple_speaker_listener_v4.env,
    "simple_reference": simple_reference_v3.env,
    "simple_adversary": simple_adversary_v3.env,
    "simple_crypto": simple_crypto_v3.env,
    "simple_push": simple_push_v3.env,
    "simple_tag": simple_tag_v3.env,
    "simple_world_comm": simple_world_comm_v3.env,
}

# Optional: SingleAgentMPEWrapper class (no changes needed here)
class SingleAgentMPEWrapper(gym.Env):
    # ... (no changes to this class) ...
    metadata = {"render_modes": []}
    def __init__(self, parallel_env):
        super().__init__()
        self.env = parallel_env
        self.possible_agents = self.env.possible_agents
        self._first = self.possible_agents[0]
        if hasattr(self.env, "observation_spaces"):
            self.observation_space = self.env.observation_spaces[self._first]
            self.action_space = self.env.action_spaces[self._first]
        else:
            self.observation_space = self.env.observation_space[self._first]
            self.action_space = self.env.action_space[self._first]
        self.max_episode_steps = getattr(self.env, "max_cycles", 25)
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return obs[self._first], infos.get(self._first, {})
    def step(self, action):
        actions = {aid: np.zeros_like(action, dtype=np.float32) for aid in self.env.agents}
        if self._first in self.env.agents: actions[self._first] = action
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        aid = self._first; done = terminations[aid] or truncations[aid]
        return next_obs[aid], rewards[aid], done, False, infos.get(aid, {})
    @property
    def _max_episode_steps(self): return self.max_episode_steps


def make_env(cfg):
    """
    Creates a PettingZoo MPE environment.
    """
    task_name = cfg.task.replace("mpe/", "")
    if task_name not in MPE_ENV_MAP:
        raise ValueError(f"Unknown MPE task: {cfg.task}. Available: {list(MPE_ENV_MAP.keys())}")

    env_fn = MPE_ENV_MAP[task_name]
    
    env_args = {
        "max_cycles": int(cfg.episode_length),
        "continuous_actions": True,
    }

    # Task-specific arguments from the config
    if task_name == "simple_spread":
        env_args["N"] = int(cfg.num_agents)
    elif task_name == "simple_adversary":
        env_args["N"] = int(cfg.num_agents) - 1
    elif task_name == "simple_tag":
        env_args["num_good"] = cfg.mpe.get("num_good", 1)
        env_args["num_adversaries"] = cfg.mpe.get("num_adversaries", 3)
        env_args["num_obstacles"] = cfg.mpe.get("num_obstacles", 2)
    elif task_name == "simple_world_comm":
        env_args["num_good"] = cfg.mpe.get("num_good", 2)
        env_args["num_adversaries"] = cfg.mpe.get("num_adversaries", 4)
        env_args["num_obstacles"] = cfg.mpe.get("num_obstacles", 1)
        
    aec_env = env_fn(**env_args)
    parallel_env = aec_to_parallel(aec_env)
    
    parallel_env.is_multi_agent = True
    parallel_env.max_cycles = int(cfg.episode_length)

    if getattr(cfg, "num_agents", 1) == 1:
        return SingleAgentMPEWrapper(parallel_env)

    return parallel_env