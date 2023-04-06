import torch
import torch.nn as nn
import torch.nn.functional as F

class PCNMetro(nn.Module):

    def __init__(self, nr_actions, nr_cells, scaling_factor, n_hidden=64):
        super(PCNMetro, self).__init__()

        self.scaling_factor = scaling_factor
        self.nr_cells = nr_cells

        # State embedding
        self.s_emb = nn.Sequential(nn.Linear(nr_cells, n_hidden),
                                   nn.Sigmoid())
        # Reward + horizon embedding
        self.c_emb = nn.Sequential(nn.Linear(3, n_hidden),
                                   nn.Sigmoid())
        # Output, fully connected layer
        self.fc = nn.Sequential(nn.Linear(n_hidden, nr_actions),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        # convert state index to one-hot encoding for Deep Sea Treasure
        state = F.one_hot(state.long(), num_classes=self.nr_cells).to(state.device).float()
        s = self.s_emb(state)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob