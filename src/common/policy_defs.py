import torch
import torch.distributions as pyd
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import copy
import abc
import math
from global_vars import device
from ..utils.layers import soft_update_params, weight_init, network_injector


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_size, action_size, capacity, device=device):
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
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if hasattr(self, "not_dones_no_max"):
            not_dones_no_max = torch.as_tensor(
                self.not_dones_no_max[idxs], device=self.device
            )
            return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

        return obses, actions, rewards, next_obses, not_dones


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        hidden_depth,
        log_std_bounds,
        network_type="linear",
    ):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = network_injector(
            obs_dim, hidden_dim, 2 * action_dim, hidden_depth, network_type
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        dist = SquashedNormal(mu, std)
        return dist


class SingleQCritic(nn.Module):
    """Critic network, single Q-learning"""

    def __init__(
        self, obs_dim, action_dim, hidden_dim, hidden_depth, network_type="linear"
    ):
        super().__init__()

        self.Q = network_injector(
            obs_dim + action_dim, hidden_dim, 1, hidden_depth, network_type
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        self.outputs["q1"] = q
        return q


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(
        self, obs_dim, action_dim, hidden_dim, hidden_depth, network_type="linear"
    ):
        super().__init__()

        self.Q1 = network_injector(
            obs_dim + action_dim, hidden_dim, 1, hidden_depth, network_type
        )
        self.Q2 = network_injector(
            obs_dim + action_dim, hidden_dim, 1, hidden_depth, network_type
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2


class GenericACAgent:
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        discount,
        actor_lr,
        critic_lr,
        critic_tau,
        batch_size,
        init_temperature=None,
        alpha_lr=None,
        target_entropy=None,
        double_critic=False,
        temperature=False,
        hidden_dim=1024,
        hidden_depth=2,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        self.double_critic = double_critic
        self.temperature = temperature

        if double_critic:
            self.critic = DoubleQCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_depth=hidden_depth,
                hidden_dim=hidden_dim,
            ).to(self.device)
            self.critic_target = DoubleQCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_depth=hidden_depth,
                hidden_dim=hidden_dim,
            ).to(self.device)
        else:
            self.critic = SingleQCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_depth=hidden_depth,
                hidden_dim=hidden_dim,
            ).to(self.device)
            self.critic_target = SingleQCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_depth=hidden_depth,
                hidden_dim=hidden_dim,
            ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_depth=hidden_depth,
            hidden_dim=hidden_dim,
            log_std_bounds=[-5, 2],
        ).to(self.device)

        if temperature:
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True

            self.target_entropy = target_entropy
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        if self.temperature:
            return self.log_alpha.exp()
        else:
            return None

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1

        # convert the action to numpy
        def to_np(t):
            if t is None:
                return None
            elif t.nelement() == 0:
                return np.array([])
            else:
                return t.cpu().detach().numpy()

        return to_np(action[0])

    def update(self, replay_buffer):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        train_batch_reward = reward.mean().item()

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        actor_loss, actor_entropy, alpha_loss = self.update_actor(obs)

        soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return train_batch_reward, critic_loss, actor_loss, actor_entropy, alpha_loss

    @abc.abstractmethod
    def update_actor(self, obs):
        pass

    @abc.abstractmethod
    def update_critic(self, obs, action, reward, next_obs, not_done_no_max):
        pass

    def save(self, save_path):
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

        if self.temperature:
            state["log_alpha"] = self.log_alpha
            state["log_alpha_optimizer"] = self.log_alpha_optimizer.state_dict()

        torch.save(state, save_path)

    def load(self, load_path):
        state = torch.load(load_path)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

        if self.temperature:
            self.log_alpha = state["log_alpha"]
            self.log_alpha_optimizer.load_state_dict(state["log_alpha_optimizer"])
