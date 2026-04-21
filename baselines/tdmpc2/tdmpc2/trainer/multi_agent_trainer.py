import copy
import torch
import numpy as np
from trainer.online_trainer import OnlineTrainer
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.ppo_agent import PPOAgent
from envs import make_env
from tensordict.tensordict import TensorDict
from collections import deque
from termcolor import colored

class MultiAgentTrainer(OnlineTrainer):
    def __init__(self, cfg, env, agent, buffer, logger):
        self.cfg = cfg
        self.env = env  # Training Env
        self.logger = logger
        
        # Create a separate environment for evaluation to avoid corrupting training state
        print("Initializing separate Evaluation Environment...")
        self.eval_env = make_env(cfg)

        # Use static list of agents
        self.agent_ids = list(getattr(env, "possible_agents", getattr(env, "agents", [])))

        self.agents = {}
        self.buffers = {}
        self.ppo_agent_data = {}
        
        # Agent Initialization
        for agent_id in self.agent_ids:
            # Determine agent type from config, default to tdmpc2
            if hasattr(self.cfg, 'agent_types') and agent_id in self.cfg.agent_types:
                agent_type = self.cfg.agent_types[agent_id]
            else:
                agent_type = 'tdmpc2'
                
            print(f"Initializing agent: {agent_id} with type: {agent_type}")

            if agent_type == 'ppo':
                agent_obs_shape = {getattr(cfg, "obs", "state"): cfg.obs_shape[agent_id]}
                num_actions = int(cfg.action_dim[agent_id])
                self.agents[agent_id] = PPOAgent(cfg, agent_obs_shape, num_actions, agent_id)
            else: # 'tdmpc2'
                self.agents[agent_id] = TDMPC2(cfg, agent_id=agent_id)
                agent_buffer_cfg = copy.deepcopy(cfg)
                obs_key = getattr(cfg, "obs", "state")
                agent_buffer_cfg.obs_shape = {obs_key: cfg.obs_shape[agent_id]}
                agent_buffer_cfg.action_dim = int(cfg.action_dim[agent_id])
                self.buffers[agent_id] = Buffer(agent_buffer_cfg)

        self._step = 0
        self._ep_idx = 0
        self._episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        # Buffers for logging
        self._good_agent_rewards_buffer = deque(maxlen=100)
        self._adversary_rewards_buffer = deque(maxlen=100)

        self._current_episode = { 
            agent_id: {"obs": [],"action": [],"reward": [],"next_obs": [],"terminated": []} 
            for agent_id in self.agent_ids if isinstance(self.agents[agent_id], TDMPC2)
        }

    def _to_torch(self, x, device=None):
        device = device or torch.device('cuda:0')
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def train(self):
        obs, infos = self.env.reset()
        t0 = {agent_id: True for agent_id in self.agent_ids}

        while self._step <= int(self.cfg.steps):
            actions = {}
            
            # --- ACTION SELECTION ---
            for agent_id in self.env.agents:
                agent = self.agents[agent_id]
                
                # if isinstance(agent, PPOAgent):
                #     action, logp, value = agent.act(obs[agent_id])
                #     one_hot_action = np.zeros(agent.num_discrete_actions, dtype=np.float32)
                #     one_hot_action[action] = 1.0
                #     actions[agent_id] = one_hot_action
                #     self.ppo_agent_data[agent_id] = (obs[agent_id], action, logp, value)
                # if isinstance(agent, PPOAgent):
                #     # act() now returns: action_squashed, action_raw, logp, value
                #     action_squashed, action_raw, logp, value = agent.act(obs[agent_id]) 
                #     actions[agent_id] = action_squashed  # Use squashed for environment
                #     # Store BOTH squashed and raw actions for buffer
                #     self.ppo_agent_data[agent_id] = (obs[agent_id], action_squashed, action_raw, logp, value)
                if isinstance(agent, PPOAgent):
                    # Added 'hxs' to the return values and 't0' to the act call
                    action_squashed, action_raw, logp, value, hxs = agent.act(obs[agent_id], t0=t0[agent_id]) 
                    actions[agent_id] = action_squashed
                    # Store the hidden state (hxs) as well so we can put it in the buffer later
                    self.ppo_agent_data[agent_id] = (obs[agent_id], action_squashed, action_raw, logp, value, hxs)
                else: # TDMPC2
                    if self._step < int(self.cfg.seed_steps):
                        act_space = self.env.action_spaces[agent_id]
                        actions[agent_id] = act_space.sample() * 2.0 - 1.0
                    else:
                        obs_tensor = self._to_torch(obs[agent_id])
                        actions[agent_id] = agent.act(
                            obs_tensor, t0=t0[agent_id], eval_mode=False
                        ).cpu().numpy()

            # --- RESCALE ACTIONS ---
            rescaled_actions = {}
            for agent_id, action in actions.items():
                # if isinstance(self.agents[agent_id], TDMPC2):
                if isinstance(self.agents[agent_id], TDMPC2) or isinstance(self.agents[agent_id], PPOAgent):
                    rescaled_actions[agent_id] = (action + 1.0) / 2.0
                else:
                    rescaled_actions[agent_id] = action

            # --- ENV STEP ---
            next_obs, rewards, terminations, truncations, infos = self.env.step(rescaled_actions)

            # --- STORAGE & LOGGING ---
            obs_key = getattr(self.cfg, "obs", "state")
            for agent_id in self.env.agents:
                agent = self.agents[agent_id]
                done = bool(terminations[agent_id] or truncations[agent_id])
                reward = rewards[agent_id]
                self._episode_rewards[agent_id] += float(reward)

                # if isinstance(agent, PPOAgent):
                #     prev_obs, act_squashed, act_raw, logp, val = self.ppo_agent_data[agent_id]
                #     agent.store_transition(prev_obs, act_squashed, act_raw, logp, reward, done, val)
                if isinstance(agent, PPOAgent):
                    # Retrieve the data we saved during the ACTION SELECTION phase
                    prev_obs, act_squashed, act_raw, logp, val, hxs = self.ppo_agent_data[agent_id]
                    # Actually store it in the buffer
                    agent.store_transition(prev_obs, act_squashed, act_raw, logp, reward, done, val, hxs)
                else: # TDMPC2
                    obs_t = self._to_torch(obs[agent_id])
                    act_t = self._to_torch(actions[agent_id])
                    rew_t = self._to_torch(rewards[agent_id])
                    next_obs_t = self._to_torch(next_obs[agent_id])
                    term_t = self._to_torch(done)

                    ep = self._current_episode[agent_id]
                    ep["obs"].append(obs_t)
                    ep["action"].append(act_t)
                    ep["reward"].append(rew_t)
                    ep["next_obs"].append(next_obs_t)
                    ep["terminated"].append(term_t)

                    max_len = int(getattr(self.cfg, "episode_length", getattr(self.cfg, "max_cycles", 25)))
                    if done or len(ep["obs"]) >= max_len:
                        T = len(ep["obs"])
                        # Handle Dict obs vs Flat obs
                        if isinstance(ep["obs"][0], dict):
                             obs_td = TensorDict({k: torch.stack([step[k] for step in ep["obs"]], dim=0) for k in ep["obs"][0].keys()}, batch_size=[T])
                             next_obs_td = TensorDict({k: torch.stack([step[k] for step in ep["next_obs"]], dim=0) for k in ep["next_obs"][0].keys()}, batch_size=[T])
                        else:
                            obs_td = TensorDict({obs_key: torch.stack(ep["obs"], dim=0)}, batch_size=[T])
                            next_obs_td = TensorDict({obs_key: torch.stack(ep["next_obs"], dim=0)}, batch_size=[T])

                        td = TensorDict({
                            "obs": obs_td,
                            "next_obs": next_obs_td,
                            "action": torch.stack(ep["action"], dim=0),
                            "reward": torch.stack(ep["reward"], dim=0),
                            "terminated": torch.stack(ep["terminated"], dim=0),
                        }, batch_size=[T])
                        self.buffers[agent_id].add(td)
                        self._current_episode[agent_id] = {"obs": [], "action": [], "reward": [], "next_obs": [], "terminated": []}

            # --- UPDATES ---
            if self._step >= int(self.cfg.seed_steps):
                for agent_id in self.agent_ids:
                    agent = self.agents[agent_id] 
                    
                    if isinstance(agent, PPOAgent):
                        if len(agent.buffer) >= self.cfg.ppo['rollout_steps']:
                            last_obs_for_ppo = next_obs.get(agent_id, obs.get(agent_id))
                            if last_obs_for_ppo is not None:
                                train_metrics = agent.update(last_obs_for_ppo)
                                if train_metrics and self._step % int(self.cfg.log_freq) == 0:
                                    self.logger.log({'step': self._step, f'{agent_id}/ppo_total_loss': train_metrics['total_loss']}, 'train')
                    
                    elif isinstance(agent, TDMPC2):
                        if hasattr(self.buffers[agent_id], '_buffer') and len(self.buffers[agent_id]._buffer) > self.cfg.batch_size:
                            train_metrics = agent.update(self.buffers[agent_id])
                            if self._step % int(self.cfg.log_freq) == 0:
                                log_data = {'step': self._step}
                                for k, v in train_metrics.items():
                                    log_data[f'{agent_id}/{k}'] = v
                                self.logger.log(log_data, 'train')

            # --- CONSOLE LOGGING ---
            if self._step > 0 and self._step % int(self.cfg.log_freq) == 0:
                print(f'Step: {self._step}', end=' ')
                if len(self._good_agent_rewards_buffer) > 0:
                    print(f'| Good Rew: {np.mean(self._good_agent_rewards_buffer):.2f}', end=' ')
                if len(self._adversary_rewards_buffer) > 0:
                    print(f'| Adv Rew: {np.mean(self._adversary_rewards_buffer):.2f}', end=' ')
                print('')

            # --- HANDLE EPISODE END ---
            if any(terminations.values()) or any(truncations.values()):
                if self._step >= int(self.cfg.seed_steps):
                    good_agent_rewards = []
                    adversary_rewards = []
                    for agent_id, total_reward in self._episode_rewards.items():
                        is_adversary = any(x in agent_id for x in ['adversary', 'eve', 'leadadversary'])
                        if is_adversary:
                            adversary_rewards.append(total_reward)
                        else:
                            good_agent_rewards.append(total_reward)
                    
                    if good_agent_rewards:
                        self._good_agent_rewards_buffer.append(np.mean(good_agent_rewards))
                    if adversary_rewards:
                        self._adversary_rewards_buffer.append(np.mean(adversary_rewards))

                obs, infos = self.env.reset()
                t0 = {agent_id: True for agent_id in self.agent_ids}
                self._episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
                self._ep_idx += 1
            else:
                obs = next_obs
                t0 = {agent_id: False for agent_id in self.agent_ids}

            # --- EVALUATION ---
            if self._step % int(self.cfg.eval_freq) == 0 and self._step >= int(self.cfg.seed_steps):
                self.evaluate()

            self._step += 1

    def evaluate(self):
        """Evaluates using the separate self.eval_env instance."""
        print(colored(f'Evaluating at step {self._step}...', 'blue'))
        eval_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # Run evaluation episodes
        for _ in range(int(self.cfg.eval_episodes)):
            # Important: Use eval_env, NOT self.env
            obs, infos = self.eval_env.reset()
            t0 = {agent_id: True for agent_id in self.agent_ids}
            episode_reward = {agent_id: 0.0 for agent_id in self.agent_ids}
            done = False

            eval_hxs = {}
            for agent_id in self.agent_ids:
                agent = self.agents[agent_id]
                if isinstance(agent, PPOAgent):
                    # Now 'agent' is defined and valid here
                    eval_hxs[agent_id] = torch.zeros(1, agent.cfg['hidden_size']).to(agent.device)

            while not done:
                actions = {}
                for agent_id in self.eval_env.agents:
                    agent = self.agents[agent_id]
                    
                    # if isinstance(agent, PPOAgent):
                    #     # eval_mode=True ensures deterministic action for PPO
                    #     action, _, _ = agent.act(obs[agent_id], eval_mode=True)
                    #     one_hot_action = np.zeros(agent.num_discrete_actions, dtype=np.float32)
                    #     one_hot_action[action] = 1.0
                    #     actions[agent_id] = one_hot_action
                    # if isinstance(agent, PPOAgent):
                    #     # eval_mode=True returns the mean of the distribution
                    #     action_squashed, _, _, _ = agent.act(obs[agent_id], eval_mode=True)
                    #     actions[agent_id] = action_squashed
                    if isinstance(agent, PPOAgent):
                        # Pass eval_hxs[agent_id] and receive the new one
                        action_squashed, _, _, _, new_hxs = agent.act(
                            obs[agent_id], 
                            t0=t0[agent_id], 
                            eval_mode=True, 
                            hxs=eval_hxs[agent_id]
                        )
                        eval_hxs[agent_id] = new_hxs # Update local eval loop state
                        actions[agent_id] = action_squashed
                    else:
                        # TDMPC2
                        actions[agent_id] = agent.act(
                            self._to_torch(obs[agent_id]), t0=t0[agent_id], eval_mode=True
                        ).cpu().numpy()
                
                # Rescale TDMPC2 actions
                rescaled_actions = {}
                for agent_id, action in actions.items():
                    if isinstance(self.agents[agent_id], TDMPC2) or isinstance(self.agents[agent_id], PPOAgent):
                        rescaled_actions[agent_id] = (action + 1.0) / 2.0
                    else:
                        rescaled_actions[agent_id] = action

                next_obs, rewards, terminations, truncations, _ = self.eval_env.step(rescaled_actions)

                for agent_id in self.eval_env.agents:
                    episode_reward[agent_id] += float(rewards[agent_id])

                if any(terminations.values()) or any(truncations.values()):
                    done = True
                else:
                    obs = next_obs
                    t0 = {agent_id: False for agent_id in self.agent_ids}

            for agent_id in self.agent_ids:
                eval_rewards[agent_id].append(episode_reward[agent_id])

        # --- LOGGING EVAL METRICS ---
        eval_metrics = {'step': self._step}
        team_rewards = {'good': [], 'adversary': []}

        for agent_id in self.agent_ids:
            mean_rew = np.mean(eval_rewards[agent_id])
            eval_metrics[f'{agent_id}/eval_reward'] = float(mean_rew)
            
            is_adversary = any(x in agent_id for x in ['adversary', 'eve', 'leadadversary'])
            if is_adversary:
                team_rewards['adversary'].append(mean_rew)
            else:
                team_rewards['good'].append(mean_rew)

        if team_rewards['good']:
            eval_metrics['team/good_eval_reward'] = float(np.mean(team_rewards['good']))
        if team_rewards['adversary']:
            eval_metrics['team/adversary_eval_reward'] = float(np.mean(team_rewards['adversary']))

        # Required for compatibility
        total_avg = float(sum(np.mean(eval_rewards[aid]) for aid in self.agent_ids))
        eval_metrics['episode_reward'] = total_avg

        self.logger.log(eval_metrics, 'eval')
        print(colored(f"Eval Result (Step {self._step}): Team Mean Rew: {total_avg:.2f}", 'green'))
        if team_rewards['good']:
            print(f"  > Good Agents: {np.mean(team_rewards['good']):.2f}")
        if team_rewards['adversary']:
            print(f"  > Adversaries: {np.mean(team_rewards['adversary']):.2f}")