import torch
from torch import nn
from torch.nn.functional import mse_loss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..utils.layers import network_injector

from ..tracking.reference_policy import TrackingReferencePolicy
from ..utils.rollout import rollout

from ..common.global_vars import TRACKING_PATH_LEN, device


class BehaviorCloningPolicy(nn.Module):
    def __init__(self):
        super(BehaviorCloningPolicy, self).__init__()
        self.layer = network_injector(5, network="linear")

    def forward(self, x):
        return self.layer(x)


class BehaviorCloningTrainer:
    def __init__(
        self,
        env,
        policy,
        dagger_iters=10,
        batch_size=TRACKING_PATH_LEN // 10,
        num_epochs=100,
    ):
        self.env = env
        self.policy = policy
        self.dagger_iters = dagger_iters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.expert_data = []
        self.generate_expert_data()

    def generate_expert_data(self):
        self.reference_policy = TrackingReferencePolicy(self.env.env.env.env.trk_params)
        self.expert_data.append(rollout(self.env, self.reference_policy))
        assert len(self.expert_data[-1]["observations"]) == TRACKING_PATH_LEN

    def load_data(self):
        x = torch.tensor(
            np.array([o for m in self.expert_data for o in m["observations"]]),
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        y = torch.tensor(
            np.array([a for m in self.expert_data for a in m["actions"]]),
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )
        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)

    def generate_new_data(self):
        result = rollout(self.env, self.policy)  # Rollout new data
        result = self.relabel_action(result)
        self.expert_data.append(result)

    def relabel_action(self, path):
        observation = path["observations"]
        expert_action = self.reference_policy(torch.from_numpy(observation))
        path["actions"] = expert_action.detach().numpy()

        return path

    def train_model(self):
        optimizer = optim.Adam(list(self.policy.parameters()), lr=1e-4)
        if self.dagger_iters < 2:
            self._train_model_(optimizer)
        else:
            for i in range(self.dagger_iters):
                self._train_model_(optimizer)
                self.generate_new_data()

        self.policy.eval()

    def _train_model_(self, optimizer):
        losses = []
        self.policy.train()
        data_loader = self.load_data()
        num_batches = len(data_loader)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for x, y in data_loader:
                optimizer.zero_grad()
                y_hat = self.policy(x)
                loss = mse_loss(y, y_hat)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # if epoch % 10 == 0:
            print("[%d] loss: %.8f" % (epoch, running_loss / num_batches))
            losses.append(loss.item())
