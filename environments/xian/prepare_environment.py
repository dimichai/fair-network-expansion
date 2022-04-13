#%%
# To run this file you should be on the root folder of the project and then run python ./environments/xian/prepare_environment.py

import sys
sys.path.append('./')

# from constants import constants
from asyncore import write
from collections import defaultdict
from pathlib import Path
# import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

price_bins = 5

if __name__ == "__main__":
    # Note: since the xian environment is already prepared, we can just initialize it.
    env_path = Path(f"./environments/xian")
    environment = Environment(env_path)


    # boundaries = torch.histogram(environment.price_mx, bins=5).bin_edges
    # buckets = torch.bucketize(environment.price_mx, boundaries)

    price_mx = environment.price_mx.clone().numpy()
    price_mx[price_mx <= 0] = np.nan
    bins = np.quantile(price_mx[~np.isnan(price_mx)], np.linspace(0, 1, price_bins + 1))[:-1]
    price_mx_binned = np.digitize(price_mx, bins).astype(np.float32)
    price_mx_binned[np.isnan(price_mx)] = np.nan

    with open(env_path / f'./price_groups_{price_bins}.txt', 'w+') as f:
        for i in range(price_mx_binned.shape[0]):
            for j in range(price_mx_binned.shape[1]):
                if not np.isnan(price_mx_binned[i, j]):
                    f.write(f'{i},{j},{price_mx_binned[i,j]}\n')

    # Calculate aggregate origin-destination flow matrix for each grid square.
    # A measure of importance of each square.
    agg_od_g = torch.zeros((environment.grid_x_size, environment.grid_y_size)).to(device)
    agg_od_v = environment.od_mx.sum(axis=1)
    # Get the grid indices.
    # TODO make it more efficient.
    # g = environment.vector_to_grid(torch.arange(0, agg_od_v.shape[0]))
    for i in range(agg_od_v.shape[0]):
        g = environment.vector_to_grid(torch.Tensor([i])).type(torch.int32)
        agg_od_g[g[0], g[1]] = agg_od_v[i]
    # Plot the aggregate od
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(agg_od_g)
    fig.suptitle(f'Xi’an, China - Aggregate OD by square')
    fig.savefig(env_path / 'aggregate_od_by_square.png')
# %%
