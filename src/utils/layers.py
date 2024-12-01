import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mlp(
    input_dim,
    hidden_dim,
    output_dim,
    hidden_depth,
    activation=nn.ReLU(inplace=True),
    output_mod=None,
):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), activation]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), activation]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class ElmanRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_depth,
        nonlinearity="tanh",
        activation=nn.functional.relu,
    ):
        super().__init__()
        self.rnn = [nn.RNN(input_size, hidden_size, nonlinearity=nonlinearity)]
        for i in range(hidden_depth - 1):
            self.rnn = [nn.RNN(hidden_size, hidden_size, nonlinearity=nonlinearity)]
        self.rnn.append(nn.RNN(hidden_size, output_size, nonlinearity=nonlinearity))
        self.activation = activation
        self.hidden_weights = [None for r in self.rnn]

    def forward(self, x):
        for i in range(len(self.rnn)):
            x, self.hidden_weights[i] = self.rnn[i](x, self.hidden_weights[i])
        return x

    def reset(self):
        self.hidden_weights = [None for r in self.rnn]


def network_injector(
    num_inputs, hidden_size=64, output_size=1, hidden_depth=2, network="rnn"
):
    if network == "rnn":
        return ElmanRNN(num_inputs, hidden_size, output_size, hidden_depth)
    return mlp(num_inputs, hidden_size, output_size, hidden_depth)


def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(20.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
