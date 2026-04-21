from copy import deepcopy
import logging
import warnings
import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
    from envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies
try:
    from envs.mujoco import make_env as make_mujoco_env
except:
    make_mujoco_env = missing_dependencies
try:
    from envs.mpe import make_env as make_mpe_env
except:
    make_mpe_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print('Creating multi-task environment with tasks:', cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError('Unknown task:', task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env
    

def make_env(cfg):
    # Quiet Gym/Gymnasium logs
    try:
        gym.logger.setLevel(logging.ERROR)
    except Exception:
        pass

    if cfg.multitask:
        env = make_multitask_env(cfg)
    else:
        env = None
        # Try MPE first, then others
        for fn in [make_mpe_env, make_dm_control_env, make_maniskill_env,
                   make_metaworld_env, make_myosuite_env, make_mujoco_env]:
            try:
                env = fn(cfg)
                if env is not None:
                    break
            except ValueError as e:
                print(f"DEBUG: Caught a ValueError in {fn.__module__}: {e}")
                pass
        if env is None:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.'
            )

        # If this is a PettingZoo parallel env (multi-agent), do NOT wrap with TensorWrapper
        if getattr(env, "is_multi_agent", False) and getattr(cfg, "num_agents", 1) > 1:
            # Fill cfg shapes/dims per agent
            if hasattr(env, "observation_spaces"):
                obs_spaces = env.observation_spaces
            elif isinstance(getattr(env, "observation_space", None), dict):
                obs_spaces = env.observation_space
            else:
                raise ValueError("Parallel PettingZoo env missing observation_spaces")

            if hasattr(env, "action_spaces"):
                act_spaces = env.action_spaces
            elif isinstance(getattr(env, "action_space", None), dict):
                act_spaces = env.action_space
            else:
                raise ValueError("Parallel PettingZoo env missing action_spaces")

            # Shapes for each agent
            cfg.obs_shape = {aid: obs_spaces[aid].shape for aid in env.possible_agents}
            cfg.action_dim = {aid: act_spaces[aid].shape[0] for aid in env.possible_agents}
            cfg.episode_length = getattr(cfg, "episode_length", getattr(env, "max_cycles", 25))
            cfg.seed_steps = max(1000, 5 * int(cfg.episode_length))
            return env
        else:
            # Single-agent: keep original TensorWrapper path
            env = TensorWrapper(env)

    # Single-agent shape filling
    try:  # Dict space
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:
        cfg.obs_shape = {getattr(cfg, 'obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * int(cfg.episode_length))
    return env