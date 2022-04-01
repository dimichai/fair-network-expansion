# Xi'an environment movement constraints.
# TODO: remove from here and integrate into Environment.py
from pathlib import Path
import numpy as np
import torch
import constants
from environment import Environment

device = constants.device

# grid limits
grid_x_size = 29
grid_y_size = 29
xian = Environment(Path('./environments/xian'))

# create an adjacency matrix of the grid squares based on all the existing lines.
# adjacency is defined as being +1 or +2 steps next to the station in the existing lines.
# o1 -- o2 -- o3 -- o4 -- o5 -> adjacent to o3 are o1, o2, o4, o5
existing_lines_adj_mx = {}
for i, line in enumerate(xian.existing_lines_full):
    for j, station in enumerate(line):
        if station not in existing_lines_adj_mx:
            existing_lines_adj_mx[station] = []
        
        if (j - 2) >= 0:
            existing_lines_adj_mx[station].append(line[j - 2])
        if (j - 1) >= 0:
            existing_lines_adj_mx[station].append(line[j - 1])
        if (j + 1) <= (len(line) - 1):
            existing_lines_adj_mx[station].append(line[j + 1])
        if (j + 2) <= (len(line) - 1):
            existing_lines_adj_mx[station].append(line[j + 2])


# define direction tensors
dir_upright = torch.tensor([-1, 1], device=device).view(1, 2)
dir_upleft = torch.tensor([-1, -1], device=device).view(1, 2)
dir_downleft = torch.tensor([1, -1], device=device).view(1, 2)
dir_downright = torch.tensor([1, 1], device=device).view(1, 2)
dir_up = torch.tensor([-1, 0], device=device).view(1, 2)
dir_right = torch.tensor([0, 1], device=device).view(1, 2)
dir_down = torch.tensor([1, 0], device=device).view(1, 2)
dir_left = torch.tensor([0, -1], device=device).view(1, 2)

# define incremental steps to select the next available grids, based on direction.
# with this setup the agent can select grids up to 2 places away.
inc_upright = torch.tensor([[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2]], device=device)
inc_upleft = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1]], device=device)
inc_downleft = torch.tensor([[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]], device=device)
inc_downright = torch.tensor([[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], device=device)
inc_up = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2]], device=device)
inc_right = torch.tensor([[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], device=device)
inc_down = torch.tensor([[0,-2],[0,-1],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]], device=device)
inc_left = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]], device=device)
inc_all = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2],
                         [1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]], device=device)

def allowed_next_vector_indices(selected_idx, selected_g_idx, last_grid, direction_vector):
    """Applies different routing constraints and returns a vector of available indices to be selected next by the agent.
    The selection is based on the last selected index and the direction of the agent.

    Args:
        selected_idx (numpy.int64): the current selected index.
        selected_g_idx (torch.Tensor): the current selected index in grid coordinates.
        last_grid (torch.Tensor): the previously selected index in grid coordinates.
        direction_vector (torch.Tensor): the current direction vector - stores all directions the agent went in the past.

    Returns:
        torch.Tensor: direction_vector: updated direction vector
        torch.Tensor: allowed_v: all available vector indices to be selected next.
    """

    # Determine the current direction of the agent, by taking it's relation to the grid versus the previous location. 
    # direction vector: [up, up-right, right, down-right, down, down-left, left, up-left]
    grid_deviation = selected_g_idx - last_grid  # CUDA  torch.Size([1, 2])  dtype: torch.int64
    grid_deviation[grid_deviation > 0] = 1
    grid_deviation[grid_deviation < 0] = -1

    if torch.equal(grid_deviation, dir_upright):
        direction_vector[0, 1] = 1
    elif torch.equal(grid_deviation, dir_upleft):
        direction_vector[0,  7] = 1
    elif torch.equal(grid_deviation, dir_downleft):
        direction_vector[0, 5] = 1
    elif torch.equal(grid_deviation, dir_downright):
        direction_vector[0, 3] = 1
    elif torch.equal(grid_deviation, dir_up):
        direction_vector[0, 0] = 1
    elif torch.equal(grid_deviation, dir_right):
        direction_vector[0, 2] = 1
    elif torch.equal(grid_deviation, dir_down):
        direction_vector[0, 4] = 1
    elif torch.equal(grid_deviation, dir_left):
        direction_vector[0, 6] = 1

    # Determine the next allowed direction, based on the last movements.
    # direction vector: [up, up-right, right, down-right, down, down-left, left, up-left]
    allowed_dir = torch.zeros((1, 8), device=device).long()
    if (direction_vector[0, 1] == 1) or (direction_vector[0, 0] == 1 and direction_vector[0, 2] == 1):
        allowed_dir[0, 1] = 1
    elif (direction_vector[0, 7] == 1) or (direction_vector[0,0] == 1 and direction_vector[0,6] == 1):
        allowed_dir[0, 7] = 1
    elif (direction_vector[0, 5] == 1) or (direction_vector[0, 4] == 1 and direction_vector[0, 6] == 1):
        allowed_dir[0, 5] = 1
    elif (direction_vector[0, 3] == 1) or (direction_vector[0, 2] == 1 and direction_vector[0, 4] == 1):
        allowed_dir[0, 3] = 1
    elif (direction_vector[0, 0] == 1) and (torch.sum(direction_vector) == 1):
        allowed_dir[0, 0] = 1
    elif (direction_vector[0, 2] == 1) and (torch.sum(direction_vector) == 1):
        allowed_dir[0, 2] = 1
    elif (direction_vector[0, 4] == 1) and (torch.sum(direction_vector) == 1):
        allowed_dir[0, 4] = 1
    elif (direction_vector[0, 6] == 1) and (torch.sum(direction_vector) == 1):
        allowed_dir[0, 6] = 1

    # Determine the grid indices allowed to be selected next, based on the calculated next allowed direction.
    if allowed_dir[0, 0] == 1:
        allowed_g = selected_g_idx + inc_up
    elif allowed_dir[0, 1] == 1:
        allowed_g = selected_g_idx + inc_upright
    elif allowed_dir[0, 2] == 1:
        allowed_g = selected_g_idx + inc_right
    elif allowed_dir[0, 3] == 1:
        allowed_g = selected_g_idx + inc_downright
    elif allowed_dir[0, 4] == 1:
        allowed_g = selected_g_idx + inc_down
    elif allowed_dir[0, 5] == 1:
        allowed_g = selected_g_idx + inc_downleft
    elif allowed_dir[0, 6] == 1:
        allowed_g = selected_g_idx + inc_left
    elif allowed_dir[0, 7] == 1:
        allowed_g = selected_g_idx + inc_upleft
    else:
        allowed_g = selected_g_idx + inc_all

    # only keep the squares that are within the grid limits.
    allowed_g = allowed_g[(allowed_g[:, 0] < grid_x_size) & (allowed_g[:, 1] < grid_y_size)]
    allowed_g = allowed_g[(allowed_g[:, 0] >= 0) & (allowed_g[:, 1] >= 0)]
    
    if allowed_g.size()[0]:
        allowed_v = xian.grid_to_vector(allowed_g)

        # TODO control this with a flag as we might not always need to apply this constraint.
        # Do not allow the agent to select an adjacent station of a station in an existing line.
        # Essentially to stop re-creating the already existing line.
        adjacent_stations = existing_lines_adj_mx.get(selected_idx)
        if adjacent_stations:
            allowed_v = allowed_v[~np.isin(allowed_v, adjacent_stations)]

        if not allowed_v.size()[0]:
            allowed_v = torch.Tensor([], device=device).long()

    else:
        allowed_v = torch.Tensor([], device=device).long()

    return direction_vector, allowed_v
