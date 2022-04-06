from environment import Environment
from os import environ
import torch

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
