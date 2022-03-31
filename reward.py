import torch


# GPU
# def reward_fn(tour_idx, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx):

#     satisfied_od_pair = satisfied_od_pair_fn(tour_idx, agent_grid_list, line_full_tensor, line_station_list, exist_line_num)
#     satisfied_od_mask = satisfied_od_mask_fn(grid_num, satisfied_od_pair)
#     satisfied_od_tensor = torch.masked_select(od_matirx, satisfied_od_mask)

#     reward = satisfied_od_tensor.sum()   # CUDA,

#     return reward

# def reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx, grid_x_max, dis_lim):

#     satisfied_od_pair = satisfied_od_pair_fn1(tour_idx_cpu, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, grid_x_max, dis_lim)
#     # up ok
#     satisfied_od_mask = satisfied_od_mask_fn1(grid_num, satisfied_od_pair)

#     satisfied_od_tensor = torch.masked_select(od_matirx, satisfied_od_mask)

#     reward = satisfied_od_tensor.sum()   # CPU

#     return reward


def od_utility(tour_idx, environment):
    sat_od = environment.satisfied_od_flows(tour_idx)
