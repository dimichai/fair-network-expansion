import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.optim as optim
from torch.functional import F
from constraints import Constraints
from environment import Environment
from pcn import PCNMetro
import constants
from mlflow import log_metric, log_artifact, log_param
import os

device = constants.device

    
# TODO: potentially move this to the environment class
def calc_segment_cost(to_grid_idx, from_grid_idx, line_unit_price):
        # this function computes the cost for building each line
        dist = to_grid_idx - from_grid_idx
        dist = dist.pow(2)
        dist = dist.sum(dim=1).float()
        dist = dist.sqrt().data.cpu().item()
        segment_cost = line_unit_price * dist
        return segment_cost

class PCNTrainer(object):
    """
    Trains the PCN model.
    """

    def __init__(self, environment: Environment, constraints: Constraints, args) -> None:
        super(PCNTrainer, self).__init__()
        
        ####### TODO SOS ####### -- determine what the scaling factors should be for the reward and horizon
        ########################################################
        ########################################################
        # check the original PCN paper.
        scaling_factor=torch.tensor([[0.1, 0.1, 0.01]]).to(device)
        ########################################################
        ########################################################

        self.environment = environment
        self.constraints = constraints
        self.model = PCNMetro(nr_actions=environment.grid_size, nr_cells=environment.grid_size, scaling_factor=scaling_factor, n_hidden=64)

    def train(self, args):
        if args.checkpoint:
            self.model.load_state_dict(torch.load(Path(args.checkpoint, 'pcn.pt'), device))
        
        now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
        save_dir = Path('./result') / f'{args.environment}_{now}'

        train_start = time.time()
        print(f'Starts training on {device} - Model location is {save_dir}')
        # Log stuff
        if not args.no_log:
            log_param('save_dir', save_dir)

            checkpoint_dir = save_dir / 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            with open(save_dir / 'args.txt', 'w') as f:
                # log the number of existing lines as a parameter.
                args_dict = vars(args)
                args_dict['existing_lines'] = len(self.environment.existing_lines)
                json.dump(args_dict, f, indent=2)

        opt = optim.Adam(self.model.parameters(), lr=args.pcn_lr)

        # Start with a random policy and generate the first experience replay buffer
            # fill buffer with random episodes
        experience_replay = []
        for _ in range(args.nr_episodes):
            transitions = []
            episode_steps = 0

            vector_index_allow = torch.tensor([1]).to(device)
            self.direction_vector = torch.zeros((1, 8), device=device).long()
            if args.budget:
                available_fund = args.budget

            mask = torch.ones(self.environment.grid_size, device=device)

            # done = False
            while True:
                episode_steps += 1
                if vector_index_allow.size()[0] == 0:
                    break
                if args.budget:
                    if available_fund <= 0:
                        break
                # Start by creating uniform probability for all actions
                action_prob = torch.ones(self.environment.grid_size).to(device) / self.environment.grid_size
                # TODO check if still valid; We add here an extra dimension to keep consistency with the NN output
                action_prob = action_prob[None, :]
                # To filter out the invalid direction, we add a large positive number to the valid directions, 
                # essentially instructing the softmax to set all other directions to zero.
                probs = F.softmax(action_prob + mask * 10000, dim=1)
                # Sample from the distribution
                action = torch.distributions.Categorical(probs).sample()

                # Get the grid index of the cell action.
                action_grid_idx = self.environment.vector_to_grid(action)
                action_grid_idx = action_grid_idx[None, :]

                # If we are on step 1 of the episode, set the selected cell to be the last_grid
                if episode_steps == 1:
                    selected_grid_cells = action_grid_idx
                    last_action_grid_idx = action_grid_idx.view(1, 2) # last action is the first action
                else:
                    last_action_grid_idx = selected_grid_cells[-1].view(1, 2) 
                    selected_grid_cells = torch.cat((selected_grid_cells, action_grid_idx), dim=0)  

                # Update feasibility rules for the next step via the constraints.
                self.direction_vector, vector_index_allow = self.constraints.allowed_vector_indices(action, action_grid_idx, last_action_grid_idx, self.direction_vector)
                
                if vector_index_allow.size()[0]: 
                    mask = self.environment.update_mask(vector_index_allow).detach()
                
                if args.budget:
                    line_cost = calc_segment_cost(action_grid_idx, last_action_grid_idx, args.line_unit_price)
                    available_fund = available_fund - line_cost - args.station_price

                print(f'Action: {action_grid_idx}, last action: {last_action_grid_idx}, segment cost {line_cost},  available fund: {available_fund}')

                #TODO HERE we need to calculate the reward.
                # At each timestep the reward of the newly added segment is being calculated,
                # and later on we will sum up the rewards of all the segments in the episode (perhaps discount as well).

                # action = np.random.randint(0, self.environment.grid_x_size)
                # n_obs, reward, done, _ = env.step(action)
                # transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
                # obs = n_obs
                
            print('----------- EPISODE ENDED, steps -----------: ', episode_steps)
            # add episode in-place
            # add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)