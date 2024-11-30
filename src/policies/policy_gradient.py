import torch
import numpy as np
import math
from torch import nn, optim

from ..utils.layers import network_injector, ElmanRNN
from ..common.global_vars import device
from ..utils.rollout import rollout


class PGPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_dim=64,
        hidden_depth=2,
        network_type="linear",
    ):
        super(PGPolicy, self).__init__()
        self.trunk = network_injector(
            num_inputs, hidden_dim, num_outputs * 2, hidden_depth, network_type
        )

    def reset(self):
        if isinstance(self.trunk, ElmanRNN):
            self.trunk.reset()

    def forward(self, x):
        outs = self.trunk(x)
        mu, std, log_std = self.dist_create(outs)
        std = torch.exp(log_std)
        action = torch.normal(mu, std)
        self.mu = mu
        return action, std, log_std

    def dist_create(self, logits):
        min_log_std = -5
        max_log_std = 5
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        loc = torch.tanh(loc)

        log_std = torch.sigmoid(scale)
        log_std = min_log_std + log_std * (max_log_std - min_log_std)
        std = torch.exp(log_std)
        return loc, std, log_std


class PGBaseline(nn.Module):
    def __init__(
        self, num_inputs, hidden_dim=64, hidden_depth=2, network_type="linear"
    ):
        super(PGBaseline, self).__init__()
        self.trunk = network_injector(
            num_inputs, hidden_dim, 1, hidden_depth, network_type
        )

    def reset(self):
        if isinstance(self.trunk, ElmanRNN):
            self.trunk.reset()

    def forward(self, x):
        v = self.trunk(x)
        return v


class PGTrainer:
    def __init__(
        self,
        env,
        policy,
        baseline,
        num_epochs=200,
        batch_size=100,
        gamma=0.99,
        baseline_batch_size=64,
        baseline_num_epochs=5,
    ):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.baseline_batch_size = baseline_batch_size
        self.baseline_num_epochs = baseline_num_epochs

        self.policy_optim = optim.Adam(self.policy.parameters())
        self.baseline_optim = optim.Adam(self.baseline.parameters())

    def train_model(self, verbose=True):
        results_rewards = []

        for iter_num in range(self.num_epochs):
            sample_trajs = []

            self.policy.reset()

            # Sampling trajectories
            for it in range(self.batch_size):
                sample_traj = rollout(self.env, self.policy)
                sample_trajs.append(sample_traj)

            # Calculate rewards
            rewards_np = np.mean(
                np.asarray([traj["rewards"].sum() for traj in sample_trajs])
            )

            # Log returns (every 10 episodes)
            if verbose and iter_num % 10 == 0:
                print("Episode: {}, reward: {}".format(iter_num, rewards_np))

            # Saving data
            results_rewards.append(rewards_np)

            # Training model
            self._train_model_(sample_trajs)

        return results_rewards

    def _train_model_(self, trajs):
        states_all = []
        actions_all = []
        returns_all = []
        for traj in trajs:
            states_singletraj = traj["observations"]
            actions_singletraj = traj["actions"]
            rewards_singletraj = traj["rewards"]
            returns_singletraj = np.zeros_like(rewards_singletraj)
            running_returns = 0
            for t in reversed(range(0, len(rewards_singletraj))):
                running_returns = rewards_singletraj[t] + self.gamma * running_returns
                returns_singletraj[t] = running_returns
            states_all.append(states_singletraj)
            actions_all.append(actions_singletraj)
            returns_all.append(returns_singletraj)
        states = np.concatenate(states_all)
        actions = np.concatenate(actions_all)
        returns = np.concatenate(returns_all)

        # Normalize the returns
        returns = (returns - returns.mean()) / returns.std() + 1e-8

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)
        for _ in range(self.baseline_num_epochs):
            np.random.shuffle(arr)
            for i in range(n // self.baseline_batch_size):
                batch_index = arr[
                    self.baseline_batch_size * i : self.baseline_batch_size * (i + 1)
                ]
                batch_index = torch.LongTensor(batch_index).to(device)
                inputs = torch.Tensor(states).to(device)[batch_index]
                target = torch.Tensor(returns).to(device)[batch_index]

                target_hat = self.baseline(inputs)
                loss = criterion(target, target_hat)

                self.baseline_optim.zero_grad()
                loss.backward()
                self.baseline_optim.step()

        _, std, logstd = self.policy(torch.Tensor(states).to(device))
        log_policy = self.log_density(
            torch.Tensor(actions).to(device), self.policy.mu, std, logstd
        )
        baseline_pred = self.baseline(torch.from_numpy(states).float().to(device))

        returns = torch.Tensor(returns).to(device)
        loss = -torch.mean(log_policy * (returns - baseline_pred))

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        del states, actions, returns, states_all, actions_all, returns_all

    def log_density(self, x, mu, std, logstd):
        var = std.pow(2)
        log_density = (
            -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
        )
        return log_density.sum(1, keepdim=True)
