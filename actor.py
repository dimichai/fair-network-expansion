# 24/03/2022: this file is just a copy of: https://github.com/weiyu123112/City-Metro-Network-Expansion-with-RL/blob/master/metro_model.py
# 24/03/2022: which is an edited copy of : https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants

device = constants.device


class DRL4Metro(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, actor, update_fn = None, mask_fn = None, v_to_g_fn = None,
                 vector_allow_fn = None):
        super(DRL4Metro, self).__init__()

        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.vector_allow_fn = vector_allow_fn
        self.v_to_g_fn = v_to_g_fn

        # Define the encoder & decoder models
        self.actor = actor

    def get_probs(self, static, dynamic):
        ptr = None
        mask = torch.ones(1, 25, device=device)
        self.actor.init_foward(static, dynamic)
        self.actor.update_dynamic(static, dynamic, ptr)
        probs = self.actor.forward(static, dynamic)
        # probs = F.softmax(probs + mask*10000, dim=1)

        return probs

    def forward(self, static, dynamic, station_num_lim, budget=None, initial_direct = None,line_unit_price = None, station_price = None,
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
            Defines the elements to consider as dynamic. For the VRP, this can be
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

        batch_size, _, sequence_size = static.size()
        self.direction_vector = torch.zeros((1, 8), device=device).long()

        if initial_direct:  # give the initial direction
            for i in initial_direct:
                self.direction_vector[0][i] = 1

        if budget:
            available_fund = budget

        if decoder_input is not None:
            self.actor.decoder_input = decoder_input

        vector_index_allow = torch.tensor([1])

        specify_original_station = 0
        # For this problem, there are no dynamic elements - so dynamic tensor is zeros.
        if dynamic.sum():
            specify_original_station = 1

            non_zero_index = torch.nonzero(dynamic)
            ptr0 = non_zero_index[0][2]
            ptr = ptr0.view(1)

            grid_index1 = self.v_to_g_fn(ptr.data[0])
            agent_current_index = ptr.data.cpu().numpy()[0]
            agent_grids = grid_index1
            exist_agent_last_grid = grid_index1.view(1, 2)  # grid_x,grid_y

            self.direction_vector, vector_index_allow = self.vector_allow_fn(agent_current_index, grid_index1,
                                                                             exist_agent_last_grid, self.direction_vector)

            if self.mask_fn is not None: 
                if vector_index_allow.size()[0]: 
                    mask = self.mask_fn(vector_index_allow).detach()
                else:
                    raise Exception('The initial station is not appropriate!!!')
            else:
                mask = torch.ones(batch_size, sequence_size, device=device)
        else:
            # Always use a mask - if no function is provided, we don't update it
            mask = torch.ones(batch_size, sequence_size, device=device)
            ptr = None

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else station_num_lim

        if specify_original_station:  # add the initial station index
            tour_idx.append(ptr.data.unsqueeze(1))

        self.actor.init_foward(static, dynamic)

        count_num = 0
        for _ in range(max_steps):
            count_num = count_num + 1

            if vector_index_allow.size()[0] == 0:
                break

            if budget:
                if available_fund <= 0:
                    break

            self.actor.update_dynamic(static, dynamic, ptr)
            probs = self.actor.forward(static, dynamic)
            probs = F.softmax(probs + mask*10000, dim=1)
            # TODO decide wether this is part of the actor architecture
            # or part of the logic

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                # print('####################  training')
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()

                logp = m.log_prob(ptr)
            else:
                # print('!!!!!!!!!!!!!!!!!!!!  Greedy')
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            # Change the vector index to grid index
            grid_index1 = self.v_to_g_fn(ptr.data[0])   # CUDA  ptr: current grid selected by network
            agent_current_index = ptr.data.cpu().numpy()[0]  # int

            # Got the agent grid index sequence
            if count_num == 1 and specify_original_station == 0:
                agent_grids = grid_index1
                exist_agent_last_grid = grid_index1.view(1, 2)  # grid_x,grid_y
            else:
                exist_agent_last_grid = agent_grids[-1].view(1, 2)
                agent_grids = torch.cat((agent_grids, grid_index1), dim=0)

            self.direction_vector, vector_index_allow = self.vector_allow_fn(agent_current_index, grid_index1, exist_agent_last_grid, self.direction_vector)

            tour_logp.append(logp.unsqueeze(1))  # logp.unsqueeze(1)
            tour_idx.append(ptr.data.unsqueeze(1))  # ptr.data.unsqueeze(1)

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, agent_current_index)   # dynamic.requires_grad = False

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                if vector_index_allow.size()[0]:
                    mask = self.mask_fn(vector_index_allow).detach()

            # budget
            if budget:
                per_line_cost = each_line_cost(grid_index1, exist_agent_last_grid)
                available_fund = available_fund - per_line_cost - station_price

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)  tour_idx.requires_grad = False
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp

class DRL4Metro_reworked(nn.Module):
    """Same as above but without directionality constraints or mask
    """

    def __init__(self, actor, update_fn = None, v_to_g_fn = None):
        super(DRL4Metro_reworked, self).__init__()

        self.update_fn = update_fn
        self.v_to_g_fn = v_to_g_fn

        # Define the encoder & decoder models
        self.actor = actor

    def get_probs(self, static, dynamic):
        ptr = None
        # mask = torch.ones(1, 25, device=device)
        self.actor.init_foward(static, dynamic)
        self.actor.update_dynamic(static, dynamic, ptr)
        probs = self.actor.forward(static, dynamic)
        # probs = F.softmax(probs + mask*10000, dim=1)

        return probs

    def forward(self, static, dynamic, station_num_lim, budget=None, initial_direct = None,line_unit_price = None, station_price = None,
                decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as dynamic. For the VRP, this can be
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

        batch_size, _, sequence_size = static.size()

        if budget:
            available_fund = budget

        if decoder_input is not None:
            self.actor.decoder_input = decoder_input

        specify_original_station = 0
        # For this problem, there are no dynamic elements - so dynamic tensor is zeros.
        if dynamic.sum():
            raise NotImplementedError("Pls go away.")
            specify_original_station = 1

            non_zero_index = torch.nonzero(dynamic)
            ptr0 = non_zero_index[0][2]
            ptr = ptr0.view(1)

            grid_index1 = self.v_to_g_fn(ptr.data[0])
            agent_current_index = ptr.data.cpu().numpy()[0]
            agent_grids = grid_index1
            exist_agent_last_grid = grid_index1.view(1, 2)  # grid_x,grid_y

            self.direction_vector, vector_index_allow = self.vector_allow_fn(agent_current_index, grid_index1,
                                                                             exist_agent_last_grid, self.direction_vector)

            if self.mask_fn is not None: 
                if vector_index_allow.size()[0]: 
                    mask = self.mask_fn(vector_index_allow).detach()
                else:
                    raise Exception('The initial station is not appropriate!!!')
            else:
                mask = torch.ones(batch_size, sequence_size, device=device)
        else:
            # Always use a mask - if no function is provided, we don't update it
            ptr = None

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = station_num_lim

        if specify_original_station:  # add the initial station index
            tour_idx.append(ptr.data.unsqueeze(1))

        self.actor.init_foward(static, dynamic)
        epsilon = 0.0001

        count_num = 0
        for _ in range(max_steps):
            count_num = count_num + 1

            if budget and available_fund <= 0:
                break

            self.actor.update_dynamic(static, dynamic, ptr)
            probs = self.actor.forward(static, dynamic)
            probs = nn.functional.softmax((probs + epsilon), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                # print('####################  training')
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()

                logp = m.log_prob(ptr)
            else:
                # print('!!!!!!!!!!!!!!!!!!!!  Greedy')
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            if ptr in tour_idx:
                break

            # After visiting a node update the dynamic representation
            # Change the vector index to grid index
            grid_index1 = self.v_to_g_fn(ptr.data[0])   # CUDA  ptr: current grid selected by network
            agent_current_index = ptr.data.cpu().numpy()[0]  # int

            # Got the agent grid index sequence
            if count_num == 1 and specify_original_station == 0:
                agent_grids = grid_index1
                exist_agent_last_grid = grid_index1.view(1, 2)  # grid_x,grid_y
            else:
                exist_agent_last_grid = agent_grids[-1].view(1, 2)
                agent_grids = torch.cat((agent_grids, grid_index1), dim=0)

            tour_logp.append(logp.unsqueeze(1))  # logp.unsqueeze(1)
            tour_idx.append(ptr.data.unsqueeze(1))  # ptr.data.unsqueeze(1)

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, agent_current_index)   # dynamic.requires_grad = False

            # budget
            if budget:
                per_line_cost = each_line_cost(grid_index1, exist_agent_last_grid)
                available_fund = available_fund - per_line_cost - station_price

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)  tour_idx.requires_grad = False
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
