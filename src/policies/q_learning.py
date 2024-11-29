import numpy as np
import copy
import torch
from torch.nn.functional import mse_loss
from agents import GenericACAgent
import global_vars

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_size, action_size, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, obs_size), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max=None):
        idxs = np.arange(self.idx, self.idx + obs.shape[0]) % self.capacity
        self.obses[idxs] = copy.deepcopy(obs)
        self.actions[idxs] = copy.deepcopy(action)
        self.rewards[idxs] = copy.deepcopy(reward)
        self.next_obses[idxs] = copy.deepcopy(next_obs)
        self.not_dones[idxs] = 1.0 - copy.deepcopy(done)
        self.not_dones_no_max[idxs] = 1.0 - copy.deepcopy(done_no_max)

        self.full = self.full or (self.idx + obs.shape[0] >= self.capacity)
        self.idx = (self.idx + obs.shape[0]) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if hasattr(self, 'not_dones_no_max'):
            not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                               device=self.device)
            return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

        return obses, actions, rewards, next_obses, not_dones


class ActorCriticAgent(GenericACAgent):
    def update_actor(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()

        q_values = self.critic(obs, action)
        actor_loss = -torch.mean(q_values)
        
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), 0, 0

    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        q_hat = self.critic(obs, action)
        q = None
        with torch.no_grad():
            next_action = self.actor(next_obs).rsample()
            next_q = self.critic_target(next_obs, next_action) if not global_vars.GLOBAL_FLAG_PROB_2 else self.critic(next_obs, next_action)
            q = reward + not_done_no_max * self.discount * next_q
        critic_loss = mse_loss(q_hat, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()
