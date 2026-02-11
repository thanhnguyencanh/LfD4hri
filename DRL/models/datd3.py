import copy
import numpy as np
import time
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_init(path):
    # dir_path = os.path.dirname(file_path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[512, 512]):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[512, 512]):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            warmup=2000,
            writer=None,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            lr=1e-4,
            hidden_sizes=[512, 512],
            normalizer=None,
            chkpt_dir=None
    ):
        self.device = device

        self.actor1 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=lr)

        self.actor2 = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
        self.actor2_target = copy.deepcopy(self.actor2)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.normalizer = normalizer
        self.chkpt_dir = chkpt_dir
        self.warmup = warmup
        self.writer = writer
        self.update_step = 0
        self.seed = 0
        path_init(self.chkpt_dir)

    def choose_action(self, state, noise_scale=0.2, validation=False):
        rng = np.random.default_rng(self.seed)  # Local RNG
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action1 = self.actor1(state)
        action2 = self.actor2(state)
        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)
        action = action1 if q1 >= q2 else action2

        if not validation:
            if self.update_step < self.warmup:
                # joint_action = torch.tensor(np.random.normal(scale=current_noise_scale, size=(self.action_dim,)), dtype=torch.float).to(device)
                action = torch.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                                      dtype=torch.float, device=device)

            else:
                current_noise_scale = noise_scale * max(0.0, 1 - self.update_step / 500000)
                action = action + torch.normal(0, current_noise_scale, size=action.shape, device=device)
            action = torch.clamp(action, -self.max_action, self.max_action)
        self.seed += 1
        return action.cpu().data.numpy().flatten()

    def learn(self, replay_buffer, batch_size=256):
        ## cross update scheme
        self.train_one_q_and_pi(replay_buffer, True, batch_size=batch_size)
        self.train_one_q_and_pi(replay_buffer, False, batch_size=batch_size)

    def train_one_q_and_pi(self, replay_buffer, update_a1=True, batch_size=256):

        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        state[:, :12] = self.normalizer.normalize(state[:, :12])
        next_state[:, :12] = self.normalizer.normalize(next_state[:, :12])
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        action = torch.FloatTensor(action).to(self.device)
        not_done = torch.FloatTensor(1 - done).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)

        with torch.no_grad():
            next_action1 = self.actor1_target(next_state)
            next_action2 = self.actor2_target(next_state)

            noise = torch.randn(
                (action.shape[0], action.shape[1]),
                dtype=action.dtype, layout=action.layout, device=action.device
            ) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action1 = (next_action1 + noise).clamp(-self.max_action, self.max_action)
            next_action2 = (next_action2 + noise).clamp(-self.max_action, self.max_action)

            next_Q1_a1 = self.critic1_target(next_state, next_action1)
            next_Q2_a1 = self.critic2_target(next_state, next_action1)

            next_Q1_a2 = self.critic1_target(next_state, next_action2)
            next_Q2_a2 = self.critic2_target(next_state, next_action2)
            ## min first, max afterward to avoid underestimation bias
            next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
            next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)

            next_Q = torch.max(next_Q1, next_Q2)
            target_Q = reward + not_done * self.discount * next_Q

        if update_a1:
            current_Q1 = self.critic1(state, action)
            critic1_loss = F.mse_loss(current_Q1, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            actor1_loss = -self.critic1(state, self.actor1(state)).mean()

            self.actor1_optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        else:
            current_Q2 = self.critic2(state, action)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            actor2_loss = -self.critic2(state, self.actor2(state)).mean()

            self.actor2_optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2_optimizer.step()

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_step += 1

    def save(self):
        torch.save(self.critic1.state_dict(), self.chkpt_dir + "/_critic1")
        torch.save(self.critic1_optimizer.state_dict(), self.chkpt_dir + "/_critic1_optimizer")
        torch.save(self.actor1.state_dict(), self.chkpt_dir + "/_actor1")
        torch.save(self.actor1_optimizer.state_dict(), self.chkpt_dir + "/_actor1_optimizer")

        torch.save(self.critic2.state_dict(), self.chkpt_dir + "/_critic2")
        torch.save(self.critic2_optimizer.state_dict(), self.chkpt_dir + "/_critic2_optimizer")
        torch.save(self.actor2.state_dict(), self.chkpt_dir + "/_actor2")
        torch.save(self.actor2_optimizer.state_dict(), self.chkpt_dir + "/_actor2_optimizer")

    def load(self):
        self.critic1.load_state_dict(torch.load(self.chkpt_dir + "/_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic1_optimizer"))
        self.actor1.load_state_dict(torch.load(self.chkpt_dir + "/_actor1"))
        self.actor1_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor1_optimizer"))

        self.critic2.load_state_dict(torch.load(self.chkpt_dir + "/_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_critic2_optimizer"))
        self.actor2.load_state_dict(torch.load(self.chkpt_dir + "/_actor2"))
        self.actor2_optimizer.load_state_dict(torch.load(self.chkpt_dir + "/_actor2_optimizer"))