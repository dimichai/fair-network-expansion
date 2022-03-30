# Xi'an environment movement constraints.
# TODO: remove from here and integrate into Environment.py
import torch
import constants

device = constants.device

# grid limits
grid_x_size = 29
grid_y_size = 29

# define direction tensors
dir_upright = torch.tensor([-1, 1]).view(1, 2).to(device) 
dir_upleft = torch.tensor([-1, -1]).view(1, 2).to(device)  
dir_downleft = torch.tensor([1, -1]).view(1, 2).to(device) 
dir_downright = torch.tensor([1, 1]).view(1, 2).to(device)  
dir_up = torch.tensor([-1, 0]).view(1, 2).to(device)  
dir_right = torch.tensor([0, 1]).view(1, 2).to(device)  
dir_down = torch.tensor([1, 0]).view(1, 2).to(device)  
dir_left = torch.tensor([0, -1]).view(1, 2).to(device)

# define incremental steps to select the next available grids, based on direction.
# with this setup the agent can select grids up to 2 places away.
inc_upright = torch.tensor([[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2]]).to(device)
inc_upleft = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1]]).to(device)
inc_downleft = torch.tensor([[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]]).to(device)
inc_downright = torch.tensor([[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).to(device)
inc_up = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2]]).to(device)
inc_right = torch.tensor([[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).to(device)
inc_down = torch.tensor([[0,-2],[0,-1],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]).to(device)
inc_left = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]]).to(device)
inc_all = torch.tensor([[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2],
                         [1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]).to(device)

def vector_allow(self, selected_idx, selected_g_idx, last_grid, direction_vector):
    """_summary_

    Args:
        selected_idx (_type_): _description_
        selected_g_idx (_type_): _description_
        last_grid (_type_): _description_
        direction_vector (torch.Tensor): the current direction vector - stores all directions the agent went in the past.

    Returns:
        _type_: _description_
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
    allowed_dir = torch.zeros((1, 8)).long().to(device)
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
        allowed_grids = selected_g_idx + inc_up
    elif allowed_dir[0, 1] == 1:
        allowed_grids = selected_g_idx + inc_upright
    elif allowed_dir[0, 2] == 1:
        allowed_grids = selected_g_idx + inc_right
    elif allowed_dir[0, 3] == 1:
        allowed_grids = selected_g_idx + inc_downright
    elif allowed_dir[0, 4] == 1:
        allowed_grids = selected_g_idx + inc_down
    elif allowed_dir[0, 5] == 1:
        allowed_grids = selected_g_idx + inc_downleft
    elif allowed_dir[0, 6] == 1:
        allowed_grids = selected_g_idx + inc_left
    elif allowed_dir[0, 7] == 1:
        allowed_grids = selected_g_idx + inc_upleft
    else:
        allowed_grids = selected_g_idx + inc_all

    # only keep the squares that are within the grid limits.
    allowed_grids = allowed_grids[(allowed_grids[:, 0] < grid_x_size) & (allowed_grids[:, 1] < grid_y_size)]
    allowed_grids = allowed_grids[(allowed_grids[:, 0] >= 0) & (allowed_grids[:, 1] >= 0)]
    
    if allowed_grids.size()[0]:
        vector_index = self.g_to_v(allowed_grids)
        vector_index_allow = self.exi_line_control(selected_idx, vector_index)

        # vector_index_allow: tensor([], device='cuda:0', dtype=torch.int64)
        if not vector_index_allow.size()[0]:
            vector_index_allow = self.null_tensor

    else:  # grids_allow =  tensor([], device='cuda:0', dtype=torch.int64)

        vector_index_allow = self.null_tensor

    return direction_vector, vector_index_allow
