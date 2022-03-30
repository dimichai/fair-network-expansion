from os import environ
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

class Trainer(object):
    """Responsible for the wholet raining process."""
    def __init__(self, environment, args):
        super(Trainer, self).__init__()
        
        # Prepare the models
        # TODO: replace allowed_next_vector_indices with environment.constraints... whatever
        self.actor = DRL4Metro(args.static_size, 
                            args.dynamic_size, 
                            args.hidden_size, 
                            args.num_layers, 
                            args.dropout, 
                            update_dynamic, 
                            environment.update_mask, 
                            v_to_g_fn=environment.vector_to_grid, 
                            vector_allow_fn=allowed_next_vector_indices).to(device)

        self.critic = StateCritic(args.static_size, args.dynamic_size, args.hidden_size).to(device)