from environment import Environment
from os import environ
import torch

from constants import device

def od_utility(tour_idx: torch.Tensor, environment: Environment):
    sat_od_mask = environment.satisfied_od_mask(tour_idx)
    reward = (environment.od_mx * sat_od_mask).sum().to(device)
    
    return reward
