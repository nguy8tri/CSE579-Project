import torch
import numpy as np
from torch import optim
from utils import log_density
from rollouts import rollout
from return_handler import return_handler


def train_model(policy, baseline, trajs, policy_optim, baseline_optim, device, gamma=0.99, baseline_train_batch_size=64,
                baseline_num_epochs=5):
    states_all = []
    actions_all = []
    returns_all = []

    # Compute returns
    for traj in trajs:
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        running_returns = 0
        for t in reversed(range(0, len(rewards_singletraj))):
            running_returns = rewards_singletraj[t] + gamma * running_returns
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
    for epoch in range(baseline_num_epochs):                       # train baseline via regression
        np.random.shuffle(arr)
        for i in range(n // baseline_train_batch_size):
            batch_index = arr[baseline_train_batch_size * i: baseline_train_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index).to(device)
            inputs = torch.Tensor(states).to(device)[batch_index]
            target = torch.Tensor(returns).to(device)[batch_index]    

            prediction = baseline(inputs)
            loss = criterion(prediction, target)

            baseline_optim.zero_grad()
            loss.backward()
            baseline_optim.step()

    action, std, logstd = policy(torch.Tensor(states).to(device))
    log_policy = log_density(torch.Tensor(actions).to(device), policy.mu, std, logstd)
    baseline_pred = baseline(torch.from_numpy(states).float().to(device))                  # compute final baseline prediction

    log_probs = log_density(torch.Tensor(actions).to(device),policy.mu, std, logstd)

    loss = (-log_probs * (torch.Tensor(returns).to(device) - baseline_pred)).mean()

    policy_optim.zero_grad()
    loss.backward()
    policy_optim.step()

    del states, actions, returns, states_all, actions_all, returns_all


# Training loop for policy gradient
def simulate_policy_pg(env, policy, baseline, num_epochs=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    
    rewards = []

    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for it in range(batch_size):
            sample_traj = rollout(
                env,
                policy,
                render=False)
            sample_trajs.append(sample_traj)

        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Record rewards for plotting
        rewards.append(np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs])))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)

    return rewards