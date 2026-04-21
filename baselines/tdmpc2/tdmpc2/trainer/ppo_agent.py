import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# ---------------------- Value/Obs Normalizer ----------------------
class RunningMeanStd:
    def __init__(self, device, insize=1):
        self.mean = torch.zeros(insize).to(device)
        self.var = torch.ones(insize).to(device)
        self.count = 1e-4

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

# ---------------------- Recurrent Buffer ----------------------
class RecurrentPPORolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, obs, action_squashed, action_raw, logp, reward, done, value, hxs):
        self.obs.append(obs)
        self.actions_raw.append(action_raw)
        self.logp.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.hxs.append(hxs) # Store hidden state

    def clear(self):
        self.obs, self.actions_raw, self.logp = [], [], []
        self.rewards, self.dones, self.values, self.hxs = [], [], [], []

    def __len__(self):
        return len(self.rewards)

    def get_batch(self):
        return (
            np.array(self.obs, dtype=np.float32),
            np.array(self.actions_raw, dtype=np.float32),
            np.array(self.logp, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.hxs, dtype=np.float32).squeeze(1)
        )

# ---------------------- Recurrent Model ----------------------
class ActorCriticRecurrent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.policy_mean = nn.Linear(hidden_size, act_dim)
        self.value = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

        # Orthogonal Initialization
        nn.init.orthogonal_(self.feature[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs, hxs, masks):
        # obs: [Batch, ObsDim] or [SeqLen, Batch, ObsDim]
        x = self.feature(obs)
        
        if x.dim() == 2: # Single Step (act)
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else: # Sequence (update)
            # Input shape: [SeqLen, Batch, Dim]
            # masks shape: [Batch, 1] (applied to initial hidden state only)
            x, hxs = self.gru(x, (hxs * masks).unsqueeze(0))
            hxs = hxs.squeeze(0)
            
        return self.policy_mean(x), torch.exp(self.log_std), self.value(x).squeeze(-1), hxs

# ---------------------- Final Agent ----------------------
class PPOAgent:
    def __init__(self, cfg, obs_shape, action_dim, agent_id):
        self.cfg = cfg.ppo 
        self.agent_id = agent_id
        self.device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cuda:0')
        
        # Correctly handle obs_shape dictionary
        obs_dim = obs_shape[list(obs_shape.keys())[0]][0] if isinstance(obs_shape, dict) else obs_shape[0]
        
        self.model = ActorCriticRecurrent(obs_dim, action_dim, self.cfg['hidden_size']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'], eps=1e-5)
        self.buffer = RecurrentPPORolloutBuffer()
        
        # [Fix #2] Input (Observation) Normalization
        self.obs_norm = RunningMeanStd(self.device, insize=obs_dim)
        
        # Value Normalization (for targets)
        self.value_norm = RunningMeanStd(self.device, insize=1)
        
        # Current hidden state for acting
        self.current_hxs = torch.zeros(1, self.cfg['hidden_size']).to(self.device)
        
        # Default parameter for chunk length if not in config
        self.chunk_len = self.cfg.get('recurrent_data_chunk_length', 10) 

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, hxs=None):
        # t0 or done mask
        mask = torch.tensor([0.0 if t0 else 1.0], device=self.device).view(1, 1)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # [Fix #2] Update Obs Norm (only in training) & Normalize
        if not eval_mode:
            self.obs_norm.update(obs_tensor.view(1, -1))
            
        # Apply normalization
        obs_norm = (obs_tensor - self.obs_norm.mean) / (torch.sqrt(self.obs_norm.var) + 1e-8)
        obs_t = obs_norm.view(1, -1)
        
        current_hxs = hxs if hxs is not None else self.current_hxs
        mu, std, val, next_hxs = self.model(obs_t, current_hxs, mask)
        dist = Normal(mu, std)
        
        raw_act = mu if eval_mode else dist.sample()
        squashed_act = torch.tanh(raw_act)
        
        logp = dist.log_prob(raw_act).sum(dim=-1)
        logp -= (2*(np.log(2) - raw_act - torch.nn.functional.softplus(-2*raw_act))).sum(dim=-1)
        
        # De-normalize value for the buffer/GAE
        val_denorm = val * torch.sqrt(self.value_norm.var) + self.value_norm.mean
        
        # ONLY UPDATE CLASS ATTRIBUTE IF NOT EVAL MODE
        if not eval_mode:
            self.current_hxs = next_hxs
            # Return old_hxs for storage
            old_hxs = current_hxs.cpu().numpy()
            return squashed_act.cpu().numpy()[0], raw_act.cpu().numpy()[0], logp.item(), val_denorm.item(), old_hxs
        else:
            # In eval, return the NEW hidden state so the loop can pass it back next time
            return squashed_act.cpu().numpy()[0], raw_act.cpu().numpy()[0], logp.item(), val_denorm.item(), next_hxs

    def store_transition(self, obs, action_squashed, action_raw, logp, reward, done, value, hxs):
        self.buffer.add(obs, action_squashed, action_raw, logp, reward, done, value, hxs)

    def update(self, last_obs):
        # # ADD THIS:
        # current_len = len(self.buffer)
        # target_len = self.cfg['rollout_steps']
        # # Print every 500 steps so we don't spam logs
        # if current_len % 2048 == 0:
        #     print(f"DEBUG: Buffer Size: {current_len} / {target_len}")

        if len(self.buffer) < self.cfg['rollout_steps']:
            return {}

        # 1. Prepare Data
        obs, acts, logp_old, rewards, dones, values, hxs = self.buffer.get_batch()
        
        # Compute GAE
        with torch.no_grad():
            last_obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device)
            # Normalize last_obs for value prediction
            last_obs_norm = (last_obs_t - self.obs_norm.mean) / (torch.sqrt(self.obs_norm.var) + 1e-8)
            last_obs_norm = last_obs_norm.view(1, -1)
            
            mask = torch.tensor([1.0], device=self.device).view(1,1)
            _, _, last_val, _ = self.model(last_obs_norm, self.current_hxs, mask)
            last_val = (last_val * torch.sqrt(self.value_norm.var) + self.value_norm.mean).item()

        # Advantage Calculation (standard GAE)
        returns = []
        advantages = []
        gae = 0.0
        v_targets = np.append(values, last_val)
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.cfg['gamma'] * v_targets[step+1] * mask - v_targets[step]
            gae = delta + self.cfg['gamma'] * self.cfg['lam'] * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v_targets[step])

        # Convert to Tensors - STRICTLY ENFORCE FLOAT32
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # [Fix #2] Normalize observations in batch
        obs_t = (obs_t - self.obs_norm.mean) / (torch.sqrt(self.obs_norm.var) + 1e-8)
        
        acts_t = torch.tensor(acts, dtype=torch.float32, device=self.device)
        logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=self.device)
        
        # These are the ones causing the crash (lists of mixed precision):
        advs_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        hxs_t = torch.tensor(hxs, dtype=torch.float32, device=self.device)
        
        # To support chunking, we need careful mask handling
        masks_t = torch.tensor(1.0 - dones, dtype=torch.float32, device=self.device).view(-1, 1)
        values_old_t = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Update Value Normalizer
        self.value_norm.update(rets_t)
        rets_norm_t = (rets_t - self.value_norm.mean) / torch.sqrt(self.value_norm.var)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        # [Fix #3] RNN Data Chunking Preparation
        # We reshape data from [T, Dim] to [ChunkLen, NumChunks, Dim]
        total_steps = len(self.buffer)
        num_chunks = total_steps // self.chunk_len
        truncate_len = num_chunks * self.chunk_len
        
        def to_chunk(t, flat=False):
            # Reshape: [Batch*ChunkLen, ...] -> [Batch, ChunkLen, ...] -> Transpose to [ChunkLen, Batch, ...]
            t_trunc = t[:truncate_len]
            if flat: # For 1D data like rewards/logp
                t_reshaped = t_trunc.view(num_chunks, self.chunk_len)
            else: # For 2D data like obs/actions
                t_reshaped = t_trunc.view(num_chunks, self.chunk_len, -1)
            
            # Swap to [SeqLen, Batch, Dim]
            return t_reshaped.transpose(0, 1)

        b_obs = to_chunk(obs_t)
        b_acts = to_chunk(acts_t)
        b_logp_old = to_chunk(logp_old_t, flat=True)
        b_advs = to_chunk(advs_t, flat=True)
        b_rets = to_chunk(rets_norm_t, flat=True)
        b_values_old = to_chunk(values_old_t, flat=True)
        
        # For Hidden states, we only need the state at the START of every chunk
        # hxs_t is [T, Hidden]. We slice with stride `chunk_len`.
        # Shape: [NumChunks, Hidden] -> unsqueeze to [1, NumChunks, Hidden] for GRU
        b_hxs_start = hxs_t[0:truncate_len:self.chunk_len].view(num_chunks, -1)
        
        # Masks for initial hidden state of each chunk
        b_masks_start = masks_t[0:truncate_len:self.chunk_len].view(num_chunks, 1)

        # 2. Training Loop (BPTT on chunks)
        # Note: We treat the "batch" as the independent chunks processed in parallel
        for _ in range(self.cfg['epochs']):
            
            # Forward pass on all chunks in parallel
            # Input: [ChunkLen, NumChunks, Dim], Hidden: [1, NumChunks, Dim]
            mu, std, values_pred, _ = self.model(b_obs, b_hxs_start, b_masks_start)
            
            # Flatten outputs back to [ChunkLen * NumChunks, ...] to calculate loss easily
            # (Or calculate loss on [ChunkLen, NumChunks])
            mu = mu.transpose(0, 1).reshape(-1, mu.shape[-1])
            std = std.transpose(0, 1).reshape(-1, std.shape[-1])
            values_pred = values_pred.transpose(0, 1).reshape(-1)
            
            # Flatten targets
            acts_flat = b_acts.transpose(0, 1).reshape(-1, b_acts.shape[-1])
            logp_old_flat = b_logp_old.transpose(0, 1).reshape(-1)
            advs_flat = b_advs.transpose(0, 1).reshape(-1)
            rets_flat = b_rets.transpose(0, 1).reshape(-1)
            values_old_flat = b_values_old.transpose(0, 1).reshape(-1)

            # Policy Loss
            dist = Normal(mu, std)
            logp = dist.log_prob(acts_flat).sum(dim=-1)
            logp -= (2*(np.log(2) - acts_flat - torch.nn.functional.softplus(-2*acts_flat))).sum(dim=-1)
            
            ratio = torch.exp(logp - logp_old_flat)
            surr1 = ratio * advs_flat
            surr2 = torch.clamp(ratio, 1.0 - self.cfg['clip_eps'], 1.0 + self.cfg['clip_eps']) * advs_flat
            pi_loss = -torch.min(surr1, surr2).mean()
            
            # [Fix #1] Clipped Value Loss
            # Use same clip epsilon as policy usually, or dedicated val_clip
            val_clip_eps = self.cfg.get('clip_eps', 0.2) 
            values_clipped = values_old_flat + (values_pred - values_old_flat).clamp(-val_clip_eps, val_clip_eps)
            
            # Use Huber loss as per Paper Table 7
            v_loss_unclipped = nn.functional.huber_loss(values_pred, rets_flat, delta=self.cfg['huber_delta'])
            # print(self.cfg['huber_delta'])
            v_loss_clipped = nn.functional.huber_loss(values_clipped, rets_flat, delta=self.cfg['huber_delta'])
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped) # Take max
            
            ent_loss = dist.entropy().sum(dim=-1).mean()
            
            loss = pi_loss + self.cfg['value_coef'] * v_loss - self.cfg['entropy_coef'] * ent_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['max_grad_norm'])
            self.optimizer.step()

        self.buffer.clear()
        return {'pi_loss': pi_loss.item(), 'v_loss': v_loss.item(), 'total_loss': loss.item()}