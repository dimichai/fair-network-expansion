#%%
# To run this file you should be on the root folder of the project and then run python ./environments/xian/prepare_environment.py
# NOTE: run this file AFTER running generate_ams_environment.py

import sys
sys.path.append('./')

# from constants import constants
from pathlib import Path
# import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from environment import Environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

house_price_bins = 5

if __name__ == "__main__":
    # Note: Run this file
    env_path = Path(f"./environments/amsterdam")
    environment = Environment(env_path)

    price_mx = environment.price_mx.cpu().clone().numpy()
    price_mx[price_mx <= 0] = np.nan

    bins = np.quantile(price_mx[~np.isnan(price_mx)], np.linspace(0, 1, house_price_bins + 1))[:-1]
    price_mx_binned = np.digitize(price_mx, bins).astype(np.float32)
    price_mx_binned[np.isnan(price_mx)] = np.nan

    with open(env_path / f'./price_groups_{house_price_bins}.txt', 'w+') as f:
        for i in range(price_mx_binned.shape[0]):
            for j in range(price_mx_binned.shape[1]):
                if not np.isnan(price_mx_binned[i, j]):
                    f.write(f'{i},{j},{price_mx_binned[i,j]}\n')

    # Plot the group membership by square.
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(price_mx_binned)
    
    values = np.unique(price_mx_binned[~np.isnan(price_mx_binned)])
    labels = list(bins)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title("Amsterdam - House price bin groups")
    plt.savefig(env_path / f'house_price_groups_{house_price_bins}.png', bbox_inches='tight')


# %% Create Amsterdam-10x10 environment, a 10x10 grid that is a subset of the original environment
# from itertools import product
# environment = Environment(Path(f"./environments/amsterdam"))
# row_indices = np.arange(12, 22)
# col_indices = np.arange(10, 20)
# # cells = np.array([[row_indices[i], col_indices[i]] for i in range(len(row_indices))])
# cells = list(product(row_indices, col_indices))
# indices = environment.grid_to_vector(np.array(cells))

# price_mx_small = environment.price_mx[row_indices[:, np.newaxis], col_indices]
# price_mx_small_binned = np.digitize(price_mx_small, bins).astype(np.float32)
# # price_mx_small_binned[np.isnan(price_mx_small)] = np.nan
# od_mx_small = environment.od_mx[indices][:, indices]
# od_mx_small = od_mx_small/od_mx_small.sum()

# fig, ax = plt.subplots(figsize=(5, 5))
# im = ax.imshow(price_mx_small_binned)
# environment.existing_lines = []
# environment.existing_lines_full = []

# # Create directory 
# env_path = Path(f"./environments/amsterdam_10x10")
# env_path.mkdir(parents=True, exist_ok=True)
# # Write the file od.txt with the OD matrix
# with open(env_path / f'./od.txt', 'w+') as f:
#     for i in range(od_mx_small.shape[0]):
#         for j in range(od_mx_small.shape[1]):
#             if od_mx_small[i, j] >= 0:
#                 f.write(f'{i},{j},{od_mx_small[i,j]}\n')
# # Write the file average_house_price_gid.txt with the average house price per group
# with open(env_path / f'./average_house_price_gid.txt', 'w+') as f:
#     for i in range(price_mx_small.shape[0]):
#         for j in range(price_mx_small.shape[1]):
#             if not np.isnan(price_mx_small[i, j]):
#                 f.write(f'{i},{j},{price_mx_small[i,j]}\n')
# # Write the file price_groups_2.txt with the group membership of each square
# with open(env_path / f'./price_groups_{house_price_bins}.txt', 'w+') as f:
#     for i in range(price_mx_small_binned.shape[0]):
#         for j in range(price_mx_small_binned.shape[1]):
#             if not np.isnan(price_mx_small_binned[i, j]):
#                 f.write(f'{i},{j},{price_mx_small_binned[i,j]}\n')
