"""
Multi-agent trainer orchestrator.

Changes for continuous action support
--------------------------------------
- _create_agents(): resolves act_dim from Box.shape[0] when continuous,
  from Discrete.n when discrete.
- __init__(): passes continuous_actions flag to create_env via cfg.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
import torch

from config import Config
from ppo_agent import PPOAgent
from mbpo_agent import MBPOAgent


class MultiAgentTrainer:
    """
    Orchestrates training for multiple agents in a PettingZoo environment.

    Responsibilities:
    - Environment interaction
    - Rollout collection
    - Agent update coordination
    - Metrics tracking and logging
    """

    def __init__(self, env, cfg: Config):
        self.env    = env
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # Get environment info
        obs_dict, _ = env.reset()
        self.agent_ids = env.agents

        # Create agents based on configuration
        self.agents = self._create_agents()

        # Training state
        self.total_steps  = 0
        self.update_count = 0

        # Episode metrics tracking
        self.episode_rewards = {
            'good':      deque(maxlen=100),
            'adversary': deque(maxlen=100),
        }
        self.current_episode_rewards = defaultdict(float)

        # Classify agents
        self.adversary_agents = [
            aid for aid in self.agent_ids
            if 'adversary' in aid or 'eve' in aid
        ]
        self.good_agents = [
            aid for aid in self.agent_ids
            if aid not in self.adversary_agents
        ]

        print(f"\n{'='*60}")
        print(f"Multi-Agent Trainer Initialized")
        print(f"{'='*60}")
        print(f"Environment:    {cfg.env_name}")
        print(f"Action Space:   {cfg.action_space_type.upper()}")
        print(f"Total Agents:   {len(self.agent_ids)}")
        print(f"Good Agents ({len(self.good_agents)}): {self.good_agents}")
        print(f"  Algorithm: {cfg.good_agent_algo.upper()}")
        print(f"Adversary Agents ({len(self.adversary_agents)}): {self.adversary_agents}")
        print(f"  Algorithm: {cfg.adversary_algo.upper()}")
        print(f"Device: {cfg.device}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _create_agents(self) -> Dict[str, Any]:
        """
        Create agents based on configuration.

        act_dim resolution:
            Discrete  : env.action_space(agent_id).n
            Continuous: env.action_space(agent_id).shape[0]
        """
        agents = {}

        for agent_id in self.agent_ids:
            obs_dim = self.env.observation_space(agent_id).shape[0]

            if self.cfg.is_continuous:
                # Continuous: Box space → shape[0] gives action dimensionality
                act_dim = self.env.action_space(agent_id).shape[0]
            else:
                # Discrete: Discrete space → .n gives number of actions
                act_dim = self.env.action_space(agent_id).n

            is_adversary = 'adversary' in agent_id.lower() or 'eve' in agent_id.lower()
            algo = self.cfg.adversary_algo if is_adversary else self.cfg.good_agent_algo

            if algo == "mbpo":
                agents[agent_id] = MBPOAgent(agent_id, obs_dim, act_dim, self.cfg)
            else:
                agents[agent_id] = PPOAgent(agent_id, obs_dim, act_dim, self.cfg)

        return agents

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        obs_dict = self.env.reset()[0] if self.total_steps == 0 else self._current_obs
        steps_collected    = 0
        episodes_completed = 0
        last_obs           = {}

        while steps_collected < num_steps:
            actions   = {}
            log_probs = {}
            values    = {}

            for agent_id, obs in obs_dict.items():
                action, log_prob, value = self.agents[agent_id].select_action(obs)
                actions[agent_id]   = action
                log_probs[agent_id] = log_prob
                values[agent_id]    = value
                last_obs[agent_id]  = obs

            next_obs_dict, rewards, dones, truncs, infos = self.env.step(actions)

            for agent_id in obs_dict.keys():
                if agent_id not in actions:
                    continue

                reward = rewards.get(agent_id, 0.0)
                done   = bool(dones.get(agent_id, False) or truncs.get(agent_id, False))

                self.current_episode_rewards[agent_id] += reward

                self.agents[agent_id].add_experience(
                    obs=obs_dict[agent_id],
                    action=actions[agent_id],
                    log_prob=log_probs[agent_id],
                    reward=reward,
                    done=done,
                    value=values[agent_id],
                )

                # MBPO replay buffer — store the squashed action (already in [0,1])
                if isinstance(self.agents[agent_id], MBPOAgent) and agent_id in next_obs_dict:
                    self.agents[agent_id].add_to_replay_buffer(
                        obs=obs_dict[agent_id],
                        action=actions[agent_id],
                        next_obs=next_obs_dict[agent_id],
                        reward=reward,
                    )

                if agent_id in next_obs_dict:
                    last_obs[agent_id] = next_obs_dict[agent_id]

            obs_dict         = next_obs_dict
            steps_collected += 1
            self.total_steps += 1

            if not obs_dict:
                good_reward = sum(
                    self.current_episode_rewards[aid] for aid in self.good_agents
                )
                adv_reward = sum(
                    self.current_episode_rewards[aid] for aid in self.adversary_agents
                )

                self.episode_rewards['good'].append(good_reward)
                self.episode_rewards['adversary'].append(adv_reward)

                self.current_episode_rewards.clear()
                episodes_completed += 1

                obs_dict, _ = self.env.reset()
                last_obs.clear()

        self._current_obs = obs_dict
        self._last_obs    = last_obs

        return {
            'steps_collected':    steps_collected,
            'episodes_completed': episodes_completed,
        }

    # ------------------------------------------------------------------
    # Agent updates
    # ------------------------------------------------------------------

    def _update_agents(self) -> Dict[str, Dict[str, float]]:
        all_metrics = {}

        for agent_id, agent in self.agents.items():
            if agent_id in self._last_obs:
                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        self._last_obs[agent_id],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                    if self.cfg.is_continuous:
                        _, _, value = agent.model(obs_tensor)
                    else:
                        _, value = agent.model(obs_tensor)

                    bootstrap_value = float(agent.denormalize_value(value.item()))
            else:
                bootstrap_value = 0.0

            metrics = agent.update(bootstrap_value)
            all_metrics[agent_id] = metrics

        self.update_count += 1
        return all_metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_updates: int = None) -> Dict[str, List[float]]:
        if num_updates is None:
            num_updates = self.cfg.max_updates

        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Total Updates: {num_updates}")
        print(f"Rollout Steps per Update: {self.cfg.rollout_steps}")
        print(f"Total Steps: {num_updates * self.cfg.rollout_steps}")
        print(f"{'='*60}\n")

        history = defaultdict(list)

        for update in range(num_updates):
            collection_metrics = self._collect_rollouts(self.cfg.rollout_steps)
            agent_metrics      = self._update_agents()

            if update % self.cfg.log_interval == 0:
                self._log_progress(update, collection_metrics, agent_metrics)

            self._store_metrics(history, collection_metrics, agent_metrics)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")

        return history

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_progress(
        self,
        update: int,
        collection_metrics: Dict[str, float],
        agent_metrics: Dict[str, Dict[str, float]],
    ):
        avg_good_reward = (
            np.mean(self.episode_rewards['good'])
            if self.episode_rewards['good'] else 0.0
        )
        avg_adv_reward = (
            np.mean(self.episode_rewards['adversary'])
            if self.episode_rewards['adversary'] else 0.0
        )

        print(f"\n{'='*60}")
        print(f"Update {update}/{self.cfg.max_updates}")
        print(f"{'='*60}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Episodes Completed: {collection_metrics['episodes_completed']}")

        print(f"\nTeam Performance (avg over last 100 episodes):")
        print(f"  Good Agents ({self.cfg.good_agent_algo.upper()}): {avg_good_reward:.2f}")
        print(f"  Adversary Agents ({self.cfg.adversary_algo.upper()}): {avg_adv_reward:.2f}")

        print(f"\nAgent Metrics:")
        for agent_id, metrics in agent_metrics.items():
            agent_type = "MBPO" if isinstance(self.agents[agent_id], MBPOAgent) else "PPO"
            print(f"  {agent_id} ({agent_type}):")

            if 'policy_loss' in metrics:
                print(f"    Policy Loss:    {metrics['policy_loss']:.4f}")
                print(f"    Value Loss:     {metrics['value_loss']:.4f}")
                print(f"    Clip Fraction:  {metrics.get('clip_fraction', 0):.4f}")

            if 'real_policy_loss' in metrics:
                print(f"    Real Policy Loss: {metrics['real_policy_loss']:.4f}")

            if 'dynamics_loss' in metrics:
                print(f"    Dynamics Loss:       {metrics['dynamics_loss']:.4f}")
                print(f"    Reward Loss:         {metrics['reward_loss']:.4f}")
                print(f"    Rollout Length:      {metrics.get('rollout_length', 0)}")
                print(f"    Imagined Transitions:{metrics.get('imagined_transitions', 0)}")
                print(f"    Replay Buffer:       {metrics.get('replay_buffer_size', 0)}")

    def _store_metrics(
        self,
        history: Dict[str, List[float]],
        collection_metrics: Dict[str, float],
        agent_metrics: Dict[str, Dict[str, float]],
    ):
        if self.episode_rewards['good']:
            history['avg_good_reward'].append(np.mean(self.episode_rewards['good']))
        if self.episode_rewards['adversary']:
            history['avg_adversary_reward'].append(np.mean(self.episode_rewards['adversary']))

        history['total_steps'].append(self.total_steps)

        for agent_id, metrics in agent_metrics.items():
            for key, value in metrics.items():
                history[f'{agent_id}_{key}'].append(value)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        print(f"\n{'='*60}")
        print(f"Evaluating for {num_episodes} episodes...")
        print(f"{'='*60}\n")

        episode_rewards = {'good': [], 'adversary': []}

        for ep in range(num_episodes):
            obs_dict, _ = self.env.reset()
            ep_rewards  = defaultdict(float)

            while obs_dict:
                actions = {}
                for agent_id, obs in obs_dict.items():
                    action, _, _ = self.agents[agent_id].select_action(
                        obs, deterministic=deterministic
                    )
                    actions[agent_id] = action

                obs_dict, rewards, dones, truncs, _ = self.env.step(actions)

                for agent_id, reward in rewards.items():
                    ep_rewards[agent_id] += reward

            good_reward = sum(ep_rewards[aid] for aid in self.good_agents)
            adv_reward  = sum(ep_rewards[aid] for aid in self.adversary_agents)

            episode_rewards['good'].append(good_reward)
            episode_rewards['adversary'].append(adv_reward)

            print(f"Episode {ep + 1}: Good={good_reward:.2f}, Adversary={adv_reward:.2f}")

        avg_metrics = {
            'avg_good_reward':      np.mean(episode_rewards['good']),
            'avg_adversary_reward': np.mean(episode_rewards['adversary']),
            'std_good_reward':      np.std(episode_rewards['good']),
            'std_adversary_reward': np.std(episode_rewards['adversary']),
        }

        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Good Agents:      {avg_metrics['avg_good_reward']:.2f} ± {avg_metrics['std_good_reward']:.2f}")
        print(f"Adversary Agents: {avg_metrics['avg_adversary_reward']:.2f} ± {avg_metrics['std_adversary_reward']:.2f}")
        print(f"{'='*60}\n")

        return avg_metrics

    # ------------------------------------------------------------------
    # Save / Load / Close
    # ------------------------------------------------------------------

    def save_agents(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        for agent_id, agent in self.agents.items():
            path = os.path.join(directory, f"{agent_id}.pt")
            agent.save(path)
            print(f"Saved {agent_id} to {path}")

    def load_agents(self, directory: str):
        import os
        for agent_id, agent in self.agents.items():
            path = os.path.join(directory, f"{agent_id}.pt")
            if os.path.exists(path):
                agent.load(path)
                print(f"Loaded {agent_id} from {path}")
            else:
                print(f"Warning: No checkpoint found for {agent_id} at {path}")

    def close(self):
        self.env.close()