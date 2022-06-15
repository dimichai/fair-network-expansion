import torch
import torch.nn as nn
import torch.nn.functional as F

import constants

device = constants.device

class Actor(nn.Module):
    # TODO write documentation
    def __init__(self, *args):
        super(Actor, self).__init__()

    def init_foward(self, static, dynamic, *args):
        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        pass

    def forward(self, static, dynamic, *args):
        raise NotImplementedError("Abstract method. This should be implemented in a child class.")

    def update_dynamic(self, static, dynamic, *args):
        pass

class PointerActor(Actor):
    def __init__(self, static_size, dynamic_size, hidden_size, num_layers, dropout):
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

        self.static_hidden = self.static_encoder(static) #static: Array of size (batch_size, feats, num_cities)
        self.dynamic_hidden = self.dynamic_encoder(dynamic)
        
        self.last_hh = torch.zeros((batch_size, self.dynamic_hidden.size()[1]),device=device,requires_grad= True)      # batch*beam x hidden_size
        self.last_cc = torch.zeros((batch_size, self.dynamic_hidden.size()[1]), device=device,requires_grad=True)

        self.update_dynamic(static, dynamic, None)
    
    def forward(self, static, dynamic, *args):
        # ... but compute a hidden rep for each element added to sequence
        decoder_hidden = self.decoder(self.decoder_input)
        # decoder_input: size (batch,static_size, 1) 
        # decoder_hidden: size  (batch, hidden_size, 1)
        decoder_hidden = torch.squeeze(decoder_hidden, 2)

        probs, self.last_hh, self.last_cc  = self.pointer(self.static_hidden,
                                        self.dynamic_hidden,
                                        decoder_hidden, self.last_hh, self.last_cc)
        #probs: size (batch,sequence_size) 
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
        self.lstm = torch.nn.LSTMCell(input_size=hidden_size, hidden_size = hidden_size, device=device)
        self.encoder_attn = Attention(hidden_size).to(device)

        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device) #conv1d_1
        

        self.project_query = nn.Linear(hidden_size, hidden_size, device=device)

        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device) #conv1d_4

        self.drop_cc = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh, last_cc):
        last_hh, last_cc = self.lstm(decoder_hidden, (last_hh, last_cc))


        last_hh = self.drop_hh(last_hh)
        last_cc = self.drop_hh(last_cc)

        static_hidden = self.project_ref(static_hidden)
        dynamic_hidden =  self.project_d(dynamic_hidden)
        last_hh_1 = self.project_query(last_hh)

        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, last_hh_1)


        return enc_attn, last_hh, last_cc