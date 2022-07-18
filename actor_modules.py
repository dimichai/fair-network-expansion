import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import constants

device = constants.device

class printModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print("x.shape:", x.shape)
        return x

class Actor(nn.Module):
    """An abstract actor architecture. Can be used as a template for new architectures,
    and provides some extra functionality."""
    def __init__(self, *args):
        super(Actor, self).__init__()

    def init_foward(self, static, dynamic, *args):
        """Function called once in the actor logic before the actor module makes multiple forward passes.
        Can be used to save computations on static grid elements."""
        pass

    def forward(self, static, dynamic, *args):
        raise NotImplementedError("Abstract method. This should be implemented in a child class.")

    def update_dynamic(self, static, dynamic, *args):
        """A functioin called every time before the actual forward pass of the actor module."""
        pass

    def construct_state(self, static, dynamic):
        """ Construct the current state s_t """
        batch_size, static_size, num_gridblocks = static.shape
        _, dynamic_size, _ = dynamic.shape

        state_l = static.view(batch_size, num_gridblocks, static_size)
        state_v = dynamic.view(batch_size, num_gridblocks, dynamic_size)
        state = torch.cat([state_l, state_v], dim=2).transpose(1, 2)
        return state

    @property
    def nr_parameters(self):
        """Return the number of trainable parameters. This gets printed at training time
        for easy model size comparison."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


class PointerActor(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, num_layers=1, dropout=0.1):
        super(PointerActor, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.static_encoder = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_encoder = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)
        self.decoder = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def init_foward(self, static, dynamic, *args):
        batch_size, _, _ = static.shape

        self.static_hidden = self.static_encoder(static)  # static: Array of size (batch_size, feats, num_cities)
        self.dynamic_hidden = self.dynamic_encoder(dynamic)

        self.last_hh = torch.zeros((batch_size, self.dynamic_hidden.size()[1]), device=device, requires_grad=True)      # batch*beam x hidden_size
        self.last_cc = torch.zeros((batch_size, self.dynamic_hidden.size()[1]), device=device, requires_grad=True)

        self.update_dynamic(static, dynamic, None)

    def forward(self, static, dynamic, *args):
        # ... but compute a hidden rep for each element added to sequence
        decoder_hidden = self.decoder(self.decoder_input)
        # decoder_input: size (batch,static_size, 1)
        # decoder_hidden: size  (batch, hidden_size, 1)
        decoder_hidden = torch.squeeze(decoder_hidden, 2)

        probs, self.last_hh, self.last_cc = self.pointer(self.static_hidden, self.dynamic_hidden,
                                                         decoder_hidden, self.last_hh, self.last_cc)
        # probs: size (batch,sequence_size)
        return probs

    def update_dynamic(self, static, dynamic, ptr, *args):
        batch_size, static_size, _ = static.shape
        self.dynamic_hidden = self.dynamic_encoder(dynamic)
        if ptr:
            self.decoder_input = torch.gather(static, 2,
                                              ptr.view(-1, 1, 1)
                                              .expand(-1, static_size, 1)).detach()
        else:
            self.decoder_input = self.x0.expand(batch_size, -1, -1)


class MLPActor(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, nr_layers=5, num_gridblocks=25):
        super(MLPActor, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        mlp = [[nn.Linear(hidden_size, hidden_size), nn.ReLU()] for i in range(nr_layers - 2)]
        mlp = [it for block in mlp for it in block]
        self.mlp = nn.Sequential(*([nn.Linear(static_size * num_gridblocks + dynamic_size * num_gridblocks, hidden_size),
                                   nn.ReLU()] +
                                   mlp +
                                   [nn.Linear(hidden_size, num_gridblocks)]))

    def forward(self, static, dynamic, *args):
        batch_size, static_size, num_gridblocks = static.shape
        _, dynamic_size, _ = dynamic.shape
        # Construct the current state s_t

        state = self.construct_state(static, dynamic)
        probs = self.mlp(state.reshape(batch_size, (num_gridblocks * static_size) + (num_gridblocks * dynamic_size)))
        return probs

class MLPActor_Attention(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, nr_layers=5, num_gridblocks=25):
        super(MLPActor_Attention, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        # Reduce size from (N^2 * (static_size + dynamic_size)) to (N^2) and project to hidden (LARGE LAYER!)
        downsample = [nn.Conv1d(static_size + dynamic_size, static_size + dynamic_size, kernel_size=1),
                      nn.ReLU(),
                      nn.Conv1d(static_size + dynamic_size, 1, kernel_size=1),
                      nn.ReLU(),
                      nn.Flatten(start_dim=1, end_dim=2),
                      nn.Linear(num_gridblocks, hidden_size),
                      nn.ReLU()]
        mlp = [[nn.Linear(hidden_size, hidden_size), nn.ReLU()] for i in range(nr_layers - 1)]
        mlp = [it for block in mlp for it in block]

        self.downsample = nn.Sequential(*downsample)
        self.mlp = nn.Sequential(*mlp)

        self.encoder_attn = Attention(hidden_size).to(device)
        self.static_hidden = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_hidden = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_uniform_(p)

    def forward(self, static, dynamic, *args):
        state = self.construct_state(static, dynamic)
        out = self.downsample(state)
        hidden = self.mlp(out)

        # Attention
        static_hidden = self.static_hidden(static)
        dynamic_hidden = self.dynamic_hidden(dynamic)
        att = self.encoder_attn(static_hidden, dynamic_hidden, hidden)
        return att


class RNNActor(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, num_gridblocks=25, *args):
        super(RNNActor, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.num_gridblocks = num_gridblocks
        self.hidden_size = hidden_size
        
        # Reduce size from (N^2 * (static_size + dynamic_size)) to (N^2)
        downsample = [nn.Conv1d(static_size + dynamic_size, static_size + dynamic_size, kernel_size=1),
                      nn.ReLU(),
                      nn.Conv1d(static_size + dynamic_size, 1, kernel_size=1),
                      nn.ReLU(),
                      nn.Flatten(start_dim=1, end_dim=2),
                      nn.Linear(num_gridblocks, hidden_size),
                      nn.ReLU()]
        self.downsample = nn.Sequential(*downsample)
        self.rnn = nn.RNN(hidden_size, hidden_size, nonlinearity="relu", batch_first=True)
        self.final = nn.Linear(hidden_size, num_gridblocks)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_uniform_(p)

    def init_foward(self, static, dynamic, *args):
        self.hidden = torch.zeros(1, self.hidden_size, requires_grad=True).to(device)

    def forward(self, static, dynamic, *args):

        # Downsample state space
        state = self.construct_state(static, dynamic)
        out = self.downsample(state).squeeze(dim=1)

        # Apply RNN
        rnn, self.hidden = self.rnn(out, self.hidden)
        if rnn.dim() == 2:
            rnn = rnn.unsqueeze(dim=1)

        # Apply attention
        last_hh = rnn[:, -1, :]
        return self.final(last_hh)


class RNNActor_Attention(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, num_gridblocks=25, *args):
        super(RNNActor_Attention, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.num_gridblocks = num_gridblocks
        self.hidden_size = hidden_size
        
        # Reduce size from (N^2 * (static_size + dynamic_size)) to (N^2)
        downsample = [nn.Conv1d(static_size + dynamic_size, static_size + dynamic_size, kernel_size=1),
                      nn.ReLU(),
                      nn.Conv1d(static_size + dynamic_size, 1, kernel_size=1),
                      nn.ReLU(),
                      nn.Flatten(start_dim=1, end_dim=2),
                      nn.Linear(num_gridblocks, hidden_size),
                      nn.ReLU()]
        self.downsample = nn.Sequential(*downsample)
        self.rnn = nn.RNN(hidden_size, hidden_size, nonlinearity="relu", batch_first=True)

        # Attention
        self.static_encoder = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dynamic_encoder = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)
        self.encoder_attn = Attention(hidden_size).to(device)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_uniform_(p)

    def init_foward(self, static, dynamic, *args):
        self.hidden = torch.zeros(1, self.hidden_size, requires_grad=True).to(device)
        self.static_hidden = self.static_encoder(static)  # static: Array of size (batch_size, feats, num_gridblocks)

    def forward(self, static, dynamic, *args):

        # Downsample state space
        state = self.construct_state(static, dynamic)
        out = self.downsample(state).squeeze(dim=1)

        # Apply RNN
        rnn, self.hidden = self.rnn(out, self.hidden)
        if rnn.dim() == 2:
            rnn = rnn.unsqueeze(dim=1)

        # Apply attention
        last_hh = rnn[:, -1, :]
        att = self.encoder_attn(self.static_hidden, self.dynamic_hidden, last_hh)
        return att

    def update_dynamic(self, static, dynamic, *args):
        self.dynamic_hidden = self.dynamic_encoder(dynamic)

class CNNActor(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, nr_layers=5, num_gridblocks=25, *args):
        super(CNNActor, self).__init__()
        self.grid_side_length = int(np.sqrt(num_gridblocks))
        assert self.grid_side_length ** 2 == num_gridblocks, "Square root error {} != {}^2. Please review this code".format(num_gridblocks, self.grid_side_length)
        assert num_gridblocks == 25, "Implementation for 5x5 grid."

        conv_l = [nn.Conv2d(static_size + dynamic_size, 16, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
                  nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
                  nn.Flatten(start_dim=1)]
        mlp = [[nn.Linear(hidden_size, hidden_size), nn.ReLU()] for _ in range(nr_layers - 1)]
        mlp = [it for block in mlp for it in block]

        self.cnn = nn.Sequential(*conv_l)
        self.mlp = nn.Sequential(*mlp, nn.Linear(hidden_size, num_gridblocks))

        for _, p in self.named_parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_uniform_(p)

    def forward(self, static, dynamic, *args):
        batch_size, static_size, _ = static.shape
        _, dynamic_size, _ = dynamic.shape

        state = self.construct_state(static, dynamic)
        cnn = self.cnn(state.view(batch_size, static_size + dynamic_size, self.grid_side_length, self.grid_side_length))
        mlp = self.mlp(cnn)

        return mlp


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.W = nn.Parameter(torch.zeros((1, hidden_size),
                                          device=device, requires_grad=True))
        self.V = nn.Parameter(torch.zeros((1, hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        decoder_hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)

        hidden = decoder_hidden + static_hidden + dynamic_hidden

        # Broadcast some dimensions so we can do batch-matrix-multiply
        W = self.W.expand(batch_size, 1, hidden_size)

        attns = torch.squeeze(torch.bmm(W, torch.tanh(hidden)), 1)

        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size

        # Used to compute a representation of the current decoder output
        self.lstm = torch.nn.LSTMCell(input_size=hidden_size, hidden_size = hidden_size, device=device)
        self.encoder_attn = Attention(hidden_size).to(device)

        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device)  # conv1d_1

        self.project_query = nn.Linear(hidden_size, hidden_size, device=device)

        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device)  # conv1d_4

        self.drop_cc = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh, last_cc):
        last_hh, last_cc = self.lstm(decoder_hidden, (last_hh, last_cc))

        last_hh = self.drop_hh(last_hh)
        last_cc = self.drop_hh(last_cc)

        static_hidden = self.project_ref(static_hidden)
        dynamic_hidden = self.project_d(dynamic_hidden)
        last_hh_1 = self.project_query(last_hh)

        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, last_hh_1)

        return enc_attn, last_hh, last_cc
