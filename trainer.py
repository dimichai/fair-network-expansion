import datetime
from os import environ
import os
from pathlib import Path
import torch
import torch.optim as optim
from actor import DRL4Metro
from critic import StateCritic
from environments.xian.constraints import allowed_next_vector_indices
import constants

device = constants.device

def update_dynamic(dynamic, current_selected_index):
    """Updates the dynamic representation of the actor to a sparse matrix with all so-far selected stations.
    Note: this does not seem to be correct. The current implementation of metro expansion does not have any 'dynamic' elements, like demand.

    Args:
        dynamic (torch.Tensor): the current dynamic matrix.
        current_selected_index (np.int64): the latest selected station.

    Returns:
        torch.Tensor: the new dynamic matrix, where all selected stations are assigned 1.
    """
    dynamic = dynamic.clone()
    dynamic[0, 0, current_selected_index] = float(1)

    return dynamic

def train(actor, critic, actor_lr, critic_lr, result_path):
    # now = '%s' % datetime.datetime.now().time().replace(':', '_')
    now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
    save_dir = result_path / now

    checkpoint_dir = save_dir / 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    average_reward_list, actor_loss_list, critic_loss_list, average_od_list, average_Ac_list = [], [], [], [], []
    # best_params = None
    best_reward = 0



class Trainer(object):
    """Responsible for the wholet raining process."""
    def __init__(self, environment, args):
        super(Trainer, self).__init__()
        
        # Prepare the models
        # TODO: replace allowed_next_vector_indices with environment.constraints... whatever
        actor = DRL4Metro(args.static_size, 
                            args.dynamic_size, 
                            args.hidden_size, 
                            args.num_layers, 
                            args.dropout, 
                            update_dynamic, 
                            environment.update_mask, 
                            v_to_g_fn=environment.vector_to_grid, 
                            vector_allow_fn=allowed_next_vector_indices).to(device)

        critic = StateCritic(args.static_size, args.dynamic_size, args.hidden_size).to(device)

        # TODO: maybe make it so that if there is a checkpoint, training log continues from that epoch and not from the start
        if args.checkpoint:
            actor.load_state_dict(torch.load(args.checkpoint / 'actor.pt', device))
            critic.load_state_dict(torch.load(args.checkpoint / 'critic.pt', device))
        
        if not args.test:
            train(actor, critic)