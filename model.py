# 14/02/2022: this file is just a copy of: https://github.com/weiyu123112/City-Metro-Network-Expansion-with-RL/blob/master/metro_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

device = constants.device

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size): 
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input): 
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len) 

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
        W = self.W.expand(batch_size, 1,hidden_size )

        attns = torch.squeeze(torch.bmm(W, torch.tanh(hidden)),1)
        return attns

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size

        # Used to compute a representation of the current decoder output
        self.lstm = torch.nn.LSTMCell(input_size=hidden_size, hidden_size = hidden_size)
        self.lstm = self.lstm.to(device)
        self.encoder_attn = Attention(hidden_size)
        self.encoder_attn = self.encoder_attn.to(device)

        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device) #conv1d_1
        

        self.project_query = nn.Linear(hidden_size, hidden_size).to(device)

        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device) #conv1d_4

        self.drop_cc = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh,last_cc):
        last_hh, last_cc = self.lstm(decoder_hidden, (last_hh,last_cc))

        last_hh = self.drop_hh(last_hh)
        last_cc = self.drop_hh(last_cc)

        static_hidden = self.project_ref(static_hidden)
        dynamic_hidden =  self.project_d(dynamic_hidden)
        last_hh_1 = self.project_query(last_hh)

        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, last_hh_1)

        return enc_attn, last_hh, last_cc

