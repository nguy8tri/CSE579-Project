from matplotlib import pyplot as plt
import torch
import numpy as np
import math
from torch import nn, optim

from ..utils.layers import network_injector, ElmanRNN
from ..common.global_vars import TRACKING_PATH_LEN, device
from ..utils.rollout import rollout


class PGPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_dim=5,
        hidden_depth=0,
        network_type="linear",
        reference_controller=None,
    ):
        super(PGPolicy, self).__init__()
        self.trunk = network_injector(
            num_inputs,
            hidden_dim,
            num_outputs * 2,
            hidden_depth,
            network_type,
            reference_controller,
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
        num_epochs=2000,
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
        results_path_len = []
        losses = 0
        losses_count = 0

        for iter_num in range(self.num_epochs):
            sample_trajs = []

            self.policy.reset()

            for param in self.policy.parameters():
                print(param)

            # Sampling trajectories
            it = 0
            skipped = 0
            while it < self.batch_size:
                sample_traj = rollout(self.env, self.policy)
                if len(sample_traj["observations"]) == TRACKING_PATH_LEN:
                    sample_trajs.append(sample_traj)
                    it += 1
                elif it < self.batch_size // 5:
                    sample_trajs.append(sample_traj)
                    it += 1
                else:
                    skipped += 1

            print(f"Trajectories Thrown Out: {skipped}")

            # Calculate rewards
            rewards_np = np.mean(
                np.asarray([traj["rewards"].mean() for traj in sample_trajs])
            )
            path_length = np.max(
                np.asarray([traj["rewards"].shape[0] for traj in sample_trajs])
            )

            # if path_length == TRACKING_PATH_LEN:
            #     for traj in sample_trajs:
            #         if len(traj["observations"][:, 0]) == TRACKING_PATH_LEN:
            #             plt.figure()
            #             plt.plot(self.env.env.env.env.reference, label="Reference")
            #             plt.plot(traj["observations"][:, 0], label="Response")
            #             plt.legend()
            #             plt.show()

            # Log returns (every episodes)
            if verbose and iter_num % 1 == 0:
                print(
                    "Episode: {}, reward: {}, max path len: {}".format(
                        iter_num, rewards_np, path_length
                    )
                )

            # Saving data
            results_rewards.append(rewards_np)
            results_path_len.append(path_length)

            # Training model
            self._train_model_(sample_trajs, losses, losses_count)

        return results_rewards, results_path_len

    def _train_model_(self, trajs, losses, losses_count):
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
        losses += loss
        losses_count += 1
        total_loss = losses / losses_count

        self.policy_optim.zero_grad()
        total_loss.backward()
        print(loss)
        self.policy_optim.step()

        del states, actions, returns, states_all, actions_all, returns_all

    def log_density(self, x, mu, std, logstd):
        var = std.pow(2)
        log_density = (
            -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
        )
        return log_density.sum(1, keepdim=True)
