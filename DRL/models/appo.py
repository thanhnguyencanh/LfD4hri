import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import shutil

def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.prob = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.prob), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, prob, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.prob.append(prob)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.prob= []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, action_dims, input_dims, alpha, fc1_dims=512, fc2_dims=512, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fc2_dims, action_dims)
        self.sigma = nn.Linear(fc2_dims, action_dims)
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        mu = self.mu(x)
        log_std = T.clamp(self.sigma(x), -20, 2)
        std = log_std.exp()
        dist = Normal(mu, std)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=512, fc2_dims=512,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, action_dims, input_dims, gamma=0.99, alpha=1e-4, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, chkpt=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(action_dims, input_dims, alpha, chkpt_dir=chkpt)
        self.critic = CriticNetwork(input_dims + 9, alpha, chkpt_dir=chkpt)
        self.memory = PPOMemory(batch_size)

    def add(self, state, action, prob, vals, reward, done):
        self.memory.store_memory(state, action, prob, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        obs = T.FloatTensor(observation['obs'].reshape(1, -1)).to(self.actor.device)
        cmb = T.FloatTensor(observation['state'].reshape(1, -1)).to(self.actor.device)  # privilege information
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(obs)
        value = self.critic(cmb)
        raw_action = dist.sample()  # reparameterized
        action = T.tanh(raw_action)
        logp = dist.log_prob(raw_action)
        logp -= T.log(1 - action.pow(2) + 1e-6)  # Jacobian correction
        logp = logp.sum(dim=-1)

        return action.cpu().numpy()[0], raw_action.cpu().numpy()[0], logp.item(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob, vals_arr, \
            reward_arr, dones_arr, batches = self.memory.generate_batches()
            # obs_np = np.stack([s["obs"] for s in self.state_arr], axis=0)  # (N, obs_dim)
            # cmb_np = np.stack([s["state"] for s in self.state_arr], axis=0)
            values = vals_arr
            advantages = np.zeros_like(reward_arr, dtype=np.float32)
            gae = 0.0

            for t in reversed(range(len(reward_arr) - 1)):
                delta = (
                        reward_arr[t]
                        + self.gamma * values[t + 1] * (1 - dones_arr[t])
                        - values[t]
                )
                gae = delta + self.gamma * self.gae_lambda * (1 - dones_arr[t]) * gae
                advantages[t] = gae

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantage = T.tensor(advantages, dtype=T.float32).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                obs_np = np.stack([state_arr[i]["obs"] for i in batch], axis=0).astype(np.float32)
                cmb_np = np.stack([state_arr[i]["state"] for i in batch], axis=0).astype(np.float32)
                states_obs = T.tensor(obs_np).to(self.actor.device)
                states_cmb = T.tensor(cmb_np).to(self.actor.device)
                logp_old = T.tensor(old_prob[batch], dtype=T.float32).to(self.actor.device)
                action_batch = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)
                # obs_np = np.stack([s["obs"] for s in states], axis=0)  # (N, obs_dim)
                # cmb_np = np.stack([s["state"] for s in states], axis=0)
                dist = self.actor(states_obs)
                critic_value = self.critic(states_cmb)

                critic_value = T.squeeze(critic_value)

                logp_new = dist.log_prob(action_batch)
                logp_new -= T.log(1 - T.tanh(action_batch).pow(2) + 1e-6)
                logp_new = logp_new.sum(dim=-1)

                logp_diff = logp_new - logp_old
                logp_diff = T.clamp(logp_diff, -20, 20)
                prob_ratio = logp_diff.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch].detach()
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

