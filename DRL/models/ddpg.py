import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import shutil

def path_init(dir_path):
    # dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def hidden_init(layer):
    """ outputs the limits for the values in the hidden layer for initialisation"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name,
                 chkpt_dir='ckpt/ddpg'):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, 1)
        self.reset_parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        path_init(self.checkpoint_file)
        self.to(self.device)

    def reset_parameters(self):
        """Reset the weights of the layers"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = T.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name,
                 chkpt_dir='ckpt/ddpg'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, n_actions)
        self.reset_parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        path_init(self.checkpoint_file)
        self.to(self.device)

    def reset_parameters(self):
        """Reset the weights of layers """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.tanh(self.fc4(x))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_best')
        T.save(self.state_dict(), checkpoint_file)

class Agent():
    def __init__(self, alpha, beta, state_dim, tau, action_dim, gamma=0.99, fc1_dims=512, fc2_dims=512, fc3_dims=128, normalizer=None,
                 chkpt=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.normalizer = normalizer
        self.warmup = 25000
        self.noise = OUActionNoise(mu=np.zeros(action_dim))

        self.actor = ActorNetwork(alpha, state_dim, fc1_dims, fc2_dims,
                                fc3_dims, n_actions=action_dim, name='actor', chkpt_dir=chkpt)
        self.critic = CriticNetwork(beta, state_dim, fc1_dims, fc2_dims,
                                fc3_dims, n_actions=action_dim, name='critic', chkpt_dir=chkpt)

        self.target_actor = ActorNetwork(alpha, state_dim, fc1_dims, fc2_dims,
                                fc3_dims, n_actions=action_dim, name='target_actor', chkpt_dir=chkpt)

        self.target_critic = CriticNetwork(beta, state_dim, fc1_dims, fc2_dims,
                                fc3_dims, n_actions=action_dim, name='target_critic', chkpt_dir=chkpt)
        self.action_dim = action_dim
        self.max_action = 1
        self.update_network_parameters(tau=1)
        self.update_step = 0
        self.seed = 0

    def choose_action(self, observation):
        # Local RNG for deterministic sequences
        rng = np.random.default_rng(self.seed)
        self.actor.eval()
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        if self.update_step < self.warmup:
            mu = T.tensor(rng.uniform(-self.max_action, self.max_action, size=(self.action_dim,)),
                          dtype=T.float32, device=self.actor.device)
        else:
            mu = self.actor.forward(state)[0]  # assuming actor returns [batch, action_dim]

        noise = T.tensor(self.noise(), dtype=T.float32, device=self.actor.device)
        mu_prime = mu + noise

        self.actor.train()
        self.seed += 1
        return mu_prime.cpu().detach().numpy()

    def save(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, replay_buffer, batch_size=256):

        states, next_states, actions, rewards, done = replay_buffer.sample(batch_size)

        states[:, :12] = self.normalizer.normalize(states[:, :12])
        next_states[:, :12] = self.normalizer.normalize(next_states[:, :12])
        states = T.tensor(states, dtype=T.float32, device=self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)

        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(next_states)
        critic_value_ = self.target_critic.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(batch_size, -1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)