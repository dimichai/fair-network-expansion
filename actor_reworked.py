# 14/02/2022: this file is just a copy of: https://github.com/weiyu123112/City-Metro-Network-Expansion-with-RL/blob/master/metro_model.py
# 24/03/2022: which is an edited copy of : https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py
# 24/03/2022: the file is re-written to ignore the dynamic elements, since the metro expansion problem has no dynamic elements.
# 24/03/2022: parking this file for now. It seems that in their paper they use the dynamic module (the selected station is the input on each t). 
#             That does not make sense but I will re-work it later.
#             For now use the actor.py file, which is a direct copy from the original.

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
        self.lstm = torch.nn.LSTMCell(input_size=hidden_size, hidden_size = hidden_size, device=device)
        self.encoder_attn = Attention(hidden_size, device=device)

        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device) #conv1d_1
        

        self.project_query = nn.Linear(hidden_size, hidden_size, device=device)

        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, device=device)

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

        return enc_attn

class MetroActor(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size, update_fn = None, mask_fn = None, v_to_g_fn = None,
                 vector_allow_fn = None, num_layers=1, dropout=0.):
        super(MetroActor, self).__init__()

        # if dynamic_size < 1:
        #     raise ValueError(':param dynamic_size: must be > 0, even if the '
        #                      'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.vector_allow_fn = vector_allow_fn
        self.v_to_g_fn = v_to_g_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        # self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        # initialise parametes with xavier initialisation
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)  

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, station_num_lim, budget=None, initial_direct=None, line_unit_price=None, station_price=None,
            decoder_input=None, last_hh=None):
        # initial_direct: direction 
        # line_unit_price: example:  1.0
        # station_price: example: 2.0
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """

        def each_line_cost(grid_index1, exist_agent_last_grid):
            # this function compute the cost for building each line
            need1 = grid_index1 - exist_agent_last_grid
            need2 = need1.pow(2)
            need3 = need2.sum(dim=1).float()
            dis = need3.sqrt().data.cpu().item()
            per_line_cost = line_unit_price * dis
            return per_line_cost


        batch_size, input_size, sequence_size = static.size()
        self.direction_vector = torch.zeros((1, 8), device=device).long()
        
        if initial_direct: # give the initial direction
            for i in initial_direct:
                self.direction_vector[0][i] = 1

        if budget:
            available_fund = budget

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1) # decoder_input size (batch,static_size,1)

        vector_index_allow = torch.tensor([1])
        
        # specify_original_station = 0
       
        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)
        
        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else station_num_lim

        # if specify_original_station:  # add the initial station index
        #     tour_idx.append(ptr.data.unsqueeze(1))

        # Static elements only need to be processed once, and can be used across all 'pointing' iterations.
        static_hidden = self.static_encoder(static) # static: Array of size (batch_size, feats, num_cities)
        # dynamic_hidden = self.dynamic_encoder(dynamic)
        
        # I don't think this is needed here - last_cc and last_hh can be None.
        # last_hh = torch.zeros((batch_size, dynamic_hidden.size()[1]), device=device, requires_grad=True)      # batch*beam x hidden_size
        # last_cc = torch.zeros((batch_size, dynamic_hidden.size()[1]), device=device, requires_grad=True)

        count_num = 0
        for _ in range(max_steps):
            count_num = count_num + 1
            
            # it might be that there is no allowed direction, therefore break.
            if vector_index_allow.size()[0] == 0:
                break  

            if budget:
                if available_fund <= 0:
                    break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)
            # decoder_input: size (batch,static_size, 1) 
            # decoder_hidden: size  (batch, hidden_size, 1)
            decoder_hidden = torch.squeeze(decoder_hidden, 2)

            probs, last_hh, last_cc  = self.pointer(static_hidden,
                                        dynamic_hidden,
                                        decoder_hidden, last_hh, last_cc)

            # It doesnt' really matter here which one to use, either will lead to 0 in the mask having 0 probability.
            #probs = F.softmax(probs + mask.log(), dim=1)      # original program
            probs = F.softmax(probs + mask*10000, dim=1)
            #probs: size (batch,sequence_size) 


            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                # print('####################  trainging')
                m = torch.distributions.Categorical(probs) 

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                
                logp = m.log_prob(ptr) #
            else:
                # print('!!!!!!!!!!!!!!!!!!!!  Greddy')
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # Change the vector index to grid index
            grid_index1 = self.v_to_g_fn(ptr.data[0])   # CUDA  ptr: current grid selected by network
            agent_current_index = ptr.data.cpu().numpy()[0] # int

            # Got the agent grid index sequence
            if count_num == 1 and specify_original_station == 0:
                agent_grids = grid_index1
                exist_agent_last_grid = grid_index1.view(1, 2)  # grid_x,grid_y
            else:
                exist_agent_last_grid = agent_grids[-1].view(1, 2) 
                agent_grids = torch.cat((agent_grids, grid_index1), dim=0)  


            self.direction_vector, vector_index_allow = self.vector_allow_fn(agent_current_index, grid_index1, exist_agent_last_grid, self.direction_vector)

            tour_logp.append(logp.unsqueeze(1)) # logp.unsqueeze(1) 
            tour_idx.append(ptr.data.unsqueeze(1)) #ptr.data.unsqueeze(1) 

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, agent_current_index)   # dynamic.requires_grad = False
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # if count_num == 1:
                #     dynamic0 = dynamic.clone()

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                if vector_index_allow.size()[0]: 
                    mask = self.mask_fn(vector_index_allow).detach()

            decoder_input = torch.gather(static, 2,
                                        ptr.view(-1, 1, 1)
                                        .expand(-1, input_size, 1)).detach()
            #decoder_input: 
            # budget
            if budget:
                per_line_cost = each_line_cost(grid_index1, exist_agent_last_grid)
                available_fund = available_fund - per_line_cost - station_price

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)  tour_idx.requires_grad = False
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp