#%% This is made as a jupyter notebook, it is made to be ran in the VSCODE interactive mode.
# But it still can be ran as a python file.
from collections import defaultdict
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import json
import pandas as pd
from environment import Environment
import numpy as np
plt.rcParams.update({'font.size': 18})
import os


#%%
def prepare_metrics_df(models: List):
    # args = defaultdict(list)
    args = pd.DataFrame()
    metrics = defaultdict(list)
    for model_path in models:
        with open(Path('result', model_path, 'result_metrics.json')) as json_file:
            data = json.load(json_file)
            metrics['model'].append(model_path)
            metrics['avg_generated_line'].append(data['avg_generated_line'])
            metrics['mean_sat_group_od'].append(data['mean_sat_group_od'])
            metrics['mean_sat_group_od_pct'].append(data['mean_sat_group_od_pct'])
            metrics['mean_sat_od_by_group'].append(data['mean_sat_od_by_group'])
            metrics['mean_sat_od_by_group_pct'].append(data['mean_sat_od_by_group_pct'])
            metrics['group_gini'].append(data['group_gini'])
            metrics['group_pct_gini'].append(data['group_pct_gini'])
            metrics['group_pct_diff'].append(1-abs(data['mean_sat_od_by_group_pct'][0] - data['mean_sat_od_by_group_pct'][1]))
            if 'mean_distance' in data:
                metrics['mean_distance'].append(data['mean_distance'])
            if 'mean_group_distance' in data:
                metrics['mean_group_distance'].append(data['mean_group_distance'])
        
        argpath = Path('result', model_path, 'args.txt')
        if argpath.is_file():
            with open(argpath) as json_file:
                data = json.load(json_file)
                data['model'] = model_path
                args = pd.concat([args, pd.DataFrame(data, index=[0])])
                # args = args.append(data, ignore_index=True)
                # for k in data.keys():
                    # args[k].append(data[k])
                    # args.append(data[k])
            
    
    df_metrics = pd.DataFrame(metrics)
    # df_args = pd.DataFrame(args)
    return pd.merge(args, df_metrics, how='right', on='model')

#%% PLOT DILEMMA RESULTS
# Paths of models to evalute
# environment = Environment(Path(f"./environments/dilemma"), groups_file='./groups.txt')
# constraints = ForwardConstraints(environment.grid_x_size, environment.grid_y_size, environment.existing_lines_full, environment.grid_to_vector)
models = ['dilemma_5x5_20220503_13_58_29.563962', 'dilemma_5x5_20220503_17_47_23.454216', 'dilemma_5x5_20220419_13_22_47.509481', 
            'dilemma_5x5_20220418_17_43_08.080415', 'dilemma_5x5_20220503_15_18_36.055557', 'dilemma_5x5_20220503_16_36_50.970871'
            , 'dilemma_5x5_20220718_12_03_53.917197']
metrics = prepare_metrics_df(models)

# To create a scatterplot with different custom markers.
# From https://github.com/matplotlib/matplotlib/issues/11155
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)

metrics_plot = metrics.drop_duplicates(['mean_sat_group_od_pct', 'group_gini'])
fig, ax = plt.subplots(figsize=(7, 7))
s = np.repeat(500, metrics_plot.shape[0])
m = ['o','^', 's', 'v']
c = ['y', '#FF99CC', '#FF0000', 'b']

scatter = mscatter(metrics_plot['mean_sat_group_od_pct'], metrics_plot['group_pct_diff'], c=c, s=s, m=m, ax=ax)
ax.set_xlabel('% of total satisfied OD flows', fontsize=18)
ax.set_ylabel('Equity of benefits (1-difference)', fontsize=18)
fig.suptitle('(B) Utility vs Equity - Dilemma Environment')
ax.set_ylim((0,1))
ax.set_xlim((0,1))

#%%
# TODO: transfer this method to the environment class.
import torch
from matplotlib import cm
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dilemma = Environment(Path(f"./environments/dilemma_5x5"))

def calculate_agg_od(environment):
    """Calculate aggregate origin-destination flow matrix for each grid square of the given environment.

    Args:
        environment (Environment): environment for which to calcualte aggregate OD per grid square.

    Returns:
        torch.Tensor: aggregate od by grid
    """
    # 
    # A measure of importance of each square.
    agg_od_g = torch.zeros((environment.grid_x_size, environment.grid_y_size)).to(device)
    agg_od_v = environment.od_mx.sum(axis=1)
    # Get the grid indices.
    for i in range(agg_od_v.shape[0]):
        g = environment.vector_to_grid(torch.Tensor([i])).type(torch.int32)
        agg_od_g[g[0], g[1]] = agg_od_v[i]

    return agg_od_g

dilemma_od = calculate_agg_od(dilemma).cpu()
fig, ax = plt.subplots(figsize=(10, 10))

im0 = ax.imshow(dilemma_od, cm.get_cmap('Blues'))
ax.set_xticks(np.arange(-.5, 4, 1))
ax.set_yticks(np.arange(-.5, 4, 1))
ax.set_xticklabels(np.arange(0, 5, 1))
ax.set_yticklabels(np.arange(0, 5, 1))
ax.grid(color='gray', linewidth=2)
# cax = fig.add_axes([0.65, 0.175, 0.2, 0.02])
fig.colorbar(im0, orientation='vertical', fraction=0.046, pad=0.04)
ax.set_title('(A) Aggregate Origin-Destination Flow')

# %% PLOT XIAN RESULTS
metrics_xian = prepare_metrics_df(['old_xian_16_01_47.060823', 'old_xian_16_03_51.724205', 'old_xian_10_48_03.743589', 'old_xian_16_22_50.580720'])
# %%
# https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged
groups_1 = metrics_xian.loc[metrics_xian['model'] == 'old_xian_16_22_50.580720'].iloc[0]['mean_sat_od_by_group_pct']
groups_2 = metrics_xian.loc[metrics_xian['model'] == 'old_xian_10_48_03.743589'].iloc[0]['mean_sat_od_by_group_pct']
ind = np.arange(5)
plt.figure(figsize=(10,7))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, groups_1 , width, label='Maximize OD', color="#71a3f7")
plt.bar(ind + width, groups_2, width, label='GGI', color="#ff6361", hatch='///')

plt.xlabel('House Price Quintiles')
plt.ylabel('% of total satisfied OD flows')
plt.title("Utility vs Equity - Xi'an Environment")
# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('1st quintile', '2nd', '3rd', '4th', '5th'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
plt.savefig('./')

# %% PLOT AMSTERDAM RESULTS
# models = ['amsterdam_20220708_11_21_23.191428']
models = []
paths  = [ f.path for f in os.scandir('./result') if f.is_dir() ]
for p in paths:
    if 'amsterdam' in p:
        models.append(p.split('/')[-1])

metrics = prepare_metrics_df(models)
metrics.to_csv('./amsterdam_results.csv', index=False)
# %%
