import torch
from torch import nn
from utils.layers import network_injector, ElmanRNN


class PGPolicy(nn.Module):
    def __init__(self, network):
        super(PGPolicy, self).__init__()
        self.trunk = network
    
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
    def __init__(self, network):
        super(PGBaseline, self).__init__()
        self.trunk = network
    
    def reset(self):
        if isinstance(self.trunk, ElmanRNN):
            self.trunk.reset()

    def forward(self, x):
        v = self.trunk(x)
        return v
