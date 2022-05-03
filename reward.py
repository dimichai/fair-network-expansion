from os import environ
from environment import Environment
import torch
from utils import gini_tensor, gini

from constants import device

def od_utility(tour_idx: torch.Tensor, environment: Environment):
    """Total sum of satisfied Origin Destination flows.

    Args:
        tour_idx (torch.Tensor): the generated line
        environment (Environment): the environment where the line is generated.

    Returns:
        torch.Tensor: sum of satisfied Origin Destination Flows.
    """
    sat_od_mask = environment.satisfied_od_mask(tour_idx)
    reward = (environment.od_mx * sat_od_mask).sum().to(device)
    
    return reward

def group_utility(tour_idx: torch.Tensor, environment: Environment, var_lambda=0, mult_gini=False):
    """Sums total satisfied Origin Destination flows of all groups 
    (equal to od_utility in cases where every square with a group also has OD flows), 
    and subtracts a lambda of the variance (to achieve minimization of differences fairness.)

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment where the line is generated.
        var_lambda (int, optional): variance weight parameter to subtract from the sum. Defaults to 0.
        mult_gini (boolean, optional): if True, it will multiply the group utility by 1-gini_index(group utility), as they do on the AI economist paper.

    Returns:
        torch.Tensor: total reward.
    """
    assert environment.group_od_mx, 'Cannot use group_utility reward without group definitions. Provide --groups_file argument'

    sat_od_mask = environment.satisfied_od_mask(tour_idx)

    sat_group_ods = torch.zeros(len(environment.group_od_mx), device=device)
    for i, g_od in enumerate(environment.group_od_mx):
        sat_group_ods[i] = (g_od * sat_od_mask).sum().item()

    if mult_gini:
        return sat_group_ods.sum() * (1 - gini(sat_group_ods.detach().cpu().numpy()))
    else:
        return sat_group_ods.sum() - var_lambda * sat_group_ods.var()


def discounted_development_utility(tour_idx: torch.Tensor, environment: Environment, p=2.0):
    """Total sum of utility as defined by City Metro Network Expansion with Reinforcement Learning paper.
   For each covered square in the generated line, calculate the distance to the other covered squares,
   discount it and multiply it with the average house price of each square.

    Args:
        tour_idx (torch.Tensor): the generated line.
        environment (Environment): the environment wher ethe line is generated.
        p (float, optional): p-norm distance to calculate: 1: manhattan, 2: euclidean, etc. Defaults to 2.0.

    Returns:
        torch.Tensor: sum of total discounted development utility
    """
    tour_idx_g = environment.vector_to_grid(tour_idx).transpose(0, 1)
    tour_ses = environment.price_mx_norm[tour_idx_g[:, 0], tour_idx_g[:, 1]]

    # total_util = torch.zeros(tour_idx_g.shape[0], device=device)
    total_util = torch.zeros(1, device=device)
    for i in range(tour_idx_g.shape[0]):
        # Calculate the distance from each origin square to every other square covered by the line.
        distance = torch.cdist(tour_idx_g[i][None, :].float(), tour_idx_g.float(), p=p).squeeze()
        # Discount squares based on their distance. (-0.5 is theoretically a tunable parameter)
        discount = torch.exp(-0.5 * distance)
        discount[i] = 0 # origin node should have no weight in the final calculation of the utility.

        # total_util[i] = (discount * tour_ses).sum()
        total_util += (discount * tour_ses).sum()

    return total_util
