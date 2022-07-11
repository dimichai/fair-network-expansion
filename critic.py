from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv


class Critic(nn.Module):
    """An abstract critic class to use a template. Also provides some extra functionality."""
    def __init__(self):
        super(Critic, self).__init__()
        pass

    def forward(self):
        raise NotImplementedError("Abstract method. Implement this!")

    @property
    def nr_parameters(self):
        """Return the number of trainable parameters. This gets printed at training time
        for easy model size comparison."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def construct_state(self, static, dynamic):
        """ Construct the current state s_t """
        batch_size, static_size, num_gridblocks = static.shape
        _, dynamic_size, _ = dynamic.shape

        state_l = static.view(batch_size, num_gridblocks, static_size)
        state_v = dynamic.view(batch_size, num_gridblocks, dynamic_size)
        state = torch.cat([state_l, state_v], dim=2).transpose(1, 2)
        return state


class PointerCritic(Critic):  # static+ dynamic + matrix present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size, grid_size):
        super(PointerCritic, self).__init__()

        self.static_encoder = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_encoder = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv2d(hidden_size * 2, 20, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(20 * grid_size, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, hidden_size, grid_x_size, grid_y_size):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        static_hidden = static_hidden.view(hidden_size, grid_x_size, grid_y_size)
        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_size, grid_y_size)

        hidden = torch.cat((static_hidden, dynamic_hidden), 0).unsqueeze(0)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        # output = self.fc4(output)
        return output


class MLPCritic(Critic):
    def __init__(self, static_size, dynamic_size, hidden_size, nr_layers=4, num_gridblocks=25):
        super(MLPCritic, self).__init__()

        # Reduce size from (N^2 * (static_size + dynamic_size)) to (N^2)
        self.cnn = nn.Sequential(*[nn.Conv1d(static_size + dynamic_size, static_size + dynamic_size, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(static_size + dynamic_size, 1, kernel_size=1)])

        mlp = [[nn.Linear(hidden_size, hidden_size), nn.ReLU()] for i in range(nr_layers - 2)]
        mlp = [it for block in mlp for it in block]
        self.mlp = nn.Sequential(*([nn.Linear(num_gridblocks, hidden_size),
                                   nn.ReLU()] +
                                   mlp +
                                   [nn.Linear(hidden_size, 1)]))

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, hidden_size, grid_x_size, grid_y_size):
        batch_size, static_size, num_gridblocks = static.shape
        _, dynamic_size, _ = dynamic.shape
        state = self.construct_state(static, dynamic)
        out = self.cnn(state)
        probs = self.mlp(out)
        return probs


class CNNCritic(Critic):
    def __init__(self, static_size, dynamic_size, hidden_size, num_gridblocks=25, *args):
        super(CNNCritic, self).__init__()
        self.grid_side_length = int(np.sqrt(num_gridblocks))
        conv_l = [nn.Conv2d(static_size + dynamic_size, 8, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3), nn.ReLU()]

        self.cnn = nn.Sequential(*conv_l)
        self.mlp = nn.Sequential(*[nn.Linear(64, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1)])

    def forward(self, static, dynamic, *args):
        batch_size, static_size, num_gridblocks = static.shape
        _, dynamic_size, _ = dynamic.shape

        state = self.construct_state(static, dynamic)
        cnn = self.cnn(state.view(batch_size, static_size + dynamic_size, self.grid_side_length, self.grid_side_length))
        mlp = self.mlp(cnn.squeeze(dim=2).squeeze(dim=2))

        return mlp
