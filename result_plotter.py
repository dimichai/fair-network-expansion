#%% This is made as a jupyter notebook, it is made to be ran in the VSCODE interactive mode.
# But it still can be ran as a python file.
from collections import defaultdict
from genericpath import isfile
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import json
import pandas as pd
from environment import Environment
import numpy as np
plt.rcParams.update({'font.size': 18})
import os
from matplotlib import cm
import torch
import numpy as np
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    if args.shape[0] > 0:
        return pd.merge(args, df_metrics, how='right', on='model')
    else:
        return df_metrics

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
xian = Environment(Path(f"./environments/xian/"), groups_file='price_groups_5.txt')
# metrics_xian = prepare_metrics_df(['old_xian_16_01_47.060823', 'old_xian_16_03_51.724205', 'old_xian_10_48_03.743589', 'old_xian_16_22_50.580720'])
metrics_xian = prepare_metrics_df(['old_xian_16_22_50.580720', 'old_xian_10_48_03.743589'])
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
# plt.savefig('./')

#%% plot lines
fig, ax = plt.subplots(figsize=(15, 10))

im1 = ax.imshow(xian.grid_groups, cm.get_cmap('viridis'))
labels = ['1st quintile', '2nd quintile', '3rd quintile', '4th quintile', '5th quintile']
values = (np.unique(xian.grid_groups[~np.isnan(xian.grid_groups)]))
colors = [ im1.cmap(im1.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ]
ax.legend(handles=patches, loc="lower right", prop={'size': 14})
ax.set_title('Generated Lines', fontsize=32)

lines = [torch.tensor([[287,315,314,313,312,311,310,339,338,367,366,395,394,393,392,391,420,419,448,477,476,505,534,563,562,561,590,589,588,587,616,645,644,643,642,671,670,699,728,786,815,814,813,812]]),
    torch.tensor([[812,756,700,644,587,588,559,530,501,502,503,504,505,476,447,418,389,360,361,332,333,334,305,306,277,278,279,250,221,222,223,224,195,196,197,168,169,170,171,172,173,144,115,86,57]])
    ]

colors = ['#71A3F7', '#FE6361']
for i, l in enumerate(lines):
    # Note here we reverse the dimensions because on scatter plots the horizontal axis is the x axis.
    l_v = xian.vector_to_grid(l).cpu()
    label = "_no_legend"
    if i == 0:
        label = "Generated Metro Lines"

    ax.plot(l_v[1], l_v[0], '-o', color=colors[i], label=label, alpha=1, markersize=12, linewidth=4)

# ax.legend()

fig.tight_layout()

# %% PLOT AMSTERDAM RESULTS
# models = ['amsterdam_20220705_18_25_09.654804', 
#     'amsterdam_20220705_18_17_31.196986', 
#     'amsterdam_20220708_11_21_23.191428',
#     'amsterdam_20220706_11_15_16.765435',
#     'amsterdam_20220807_22_41_55.956804']
# model_types = ['ses_1', 'ses_0', 'ggi_2', 'var_3', 'rawls']
# amsterdam_20220706_14_01_39.431117  -- Rawls
# 
# models = ['amsterdam_20220706_14_01_39.431117']
import matplotlib.patches as mpatches

amsterdam = Environment(Path(f"./environments/amsterdam/"), groups_file='price_groups_5.txt')
models = []
paths  = [ f.path for f in os.scandir('./result') if f.is_dir() ]
for p in paths:
    if 'amsterdam' in p:
        models.append(p.split('/')[-1])

metrics_ams = prepare_metrics_df(models)
metrics_ams['lowest_quintile_sat_od_pct'] = metrics_ams['mean_sat_od_by_group_pct'].str[0]

#%%
# fig, ax = plt.subplots(figsize=(15, 10))

# im1 = ax.imshow(amsterdam.grid_groups, cm.get_cmap('viridis'))
# labels = ['1st quintile', '2nd quintile', '3rd quintile', '4th quintile', '5th quintile']
# values = (np.unique(amsterdam.grid_groups[~np.isnan(amsterdam.grid_groups)]))
# colors = [ im1.cmap(im1.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ]
# ax.legend(handles=patches, loc="lower left", prop={'size': 14})
# ax.set_title('(B) House Price Index Quintiles')

# # for i, l in enumerate(metrics['avg_generated_line'][0]):
# for i, l in enumerate(metrics['avg_generated_line'].tolist()):
#     # Note here we reverse the dimensions because on scatter plots the horizontal axis is the x axis.
#     label = "_no_legend"
#     if i == 0:
#         label = "Generated Metro Lines"

#     ax.plot(l[1], l[0], '-ok', label=label)

# ax.legend()

# fig.tight_layout()
# # metrics.to_csv('./amsterdam_results.csv', index=False)
# # %%

#%% Utility vs Equity - Amsterdam Environment - NOT DONE
def bar_plot(model_names, model_types, model_colors, model_hatches, metrics_df: pd.DataFrame, env: str):
    # Width of a bar 
    width = 0.3      
    ind = np.arange(5)
    xpos = ind
    fig, ax = plt.subplots(figsize=(10,7))
    for i, model in enumerate(model_names):
        results = metrics_df.loc[metrics_df['model'] == model].iloc[0]['mean_sat_od_by_group_pct']
        # position of the bar on the x axis
        # xpos = ind if i == 0 else ind + width
        ax.bar(xpos, results , width, label=model_types[i], color=model_colors[i], hatch=model_hatches[i])
        xpos = xpos + width

    plt.xlabel('House Price Quintiles')
    plt.ylabel('% of total satisfied OD flows')
    plt.title(f"Utility vs Equity - {env} Environment")
    plt.xticks(ind + width / 2, ('1st quintile', '2nd', '3rd', '4th', '5th'))

    fig.legend()
    # return fig

ams_colors = ['#71a3f7', '#ff6361', '#932DFB', 'gray', 'black']
fig = bar_plot(models[:3], model_types[:3], ams_colors, ['', '///', '+'], metrics_ams, 'Amsterdam')
#%%
groups_1 = metrics_ams.loc[metrics_ams['model'] == models[0]].iloc[0]['mean_sat_od_by_group_pct']
groups_2 = metrics_ams.loc[metrics_ams['model'] == models[1]].iloc[0]['mean_sat_od_by_group_pct']
ind = np.arange(5)
plt.figure(figsize=(10,7))

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, groups_1 , width, label=model_types[0], color="#71a3f7")
plt.bar(ind + width, groups_2, width, label=model_types[1], color="#ff6361", hatch='///')

plt.xlabel('House Price Quintiles')
plt.ylabel('% of total satisfied OD flows')
plt.title("Utility vs Equity - Amsterdam Environment")
# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
plt.xticks(ind + width / 2, ('1st quintile', '2nd', '3rd', '4th', '5th'))

# Finding the best position for legends and putting it
plt.legend(loc='best')
plt.show()
# %%
def plot_lines(env: Environment, metrics_df, line_colors: list, lines=None, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(15, 10))    
    im1 = ax.imshow(env.grid_groups, cm.get_cmap('viridis'), alpha=0.3)
    labels = ['1st quintile', '2nd quintile', '3rd quintile', '4th quintile', '5th quintile']
    values = (np.unique(env.grid_groups[~np.isnan(env.grid_groups)]))
    grid_colors = [ im1.cmap(im1.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=grid_colors[i], label=labels[i] ) for i in range(len(labels)) ]

    # TODO check how to add a line patch here.
    patches.append(mpatches.Patch(color=line_colors[0], label=model_types[0]))
    
    ax.legend(handles=patches, loc=legend_loc, prop={'size': 14})
    ax.set_title('Generated Lines', fontsize=32)
    if lines:
        for i, l in enumerate(lines):
            # Note here we reverse the dimensions because on scatter plots the horizontal axis is the x axis.
            l_v = env.vector_to_grid(l).cpu()
            label = model_types[i]
            # label = "_no_legend"
            # if i == 0:
                # label = "Generated Metro Lines"

            ax.plot(l_v[1], l_v[0], '-o', color=line_colors[i], label=label, alpha=0.8, markersize=12, linewidth=4)
    else:
        # for i, l in enumerate(metrics_df['avg_generated_line']):
        # BROKEN
        for i, model in enumerate(metrics_df['model']):
            with open(Path('result', model, 'tour_idx_multiple.txt')) as f:
                l = [int(idx) for idx in f.readline().split(',')]
                l_v = env.vector_to_grid(torch.tensor(l).reshape(-1,1)).cpu()

                label = "_no_legend"
                if i == 0:
                    label = "Generated Metro Lines"

            ax.plot(l_v[:, 1], l_v[:, 0], '-o', color=line_colors[i], label=label, alpha=1, markersize=12, linewidth=4)

    fig.tight_layout()
    return fig

# Plot Amsterdam Lines
lines = [torch.tensor([[1097,1050,1051,1004,1005,958,959,912,865,866,819,772,773,726,727,728,681,682,635,636]]),
    torch.tensor([[436,532,533,580,627,674,675,722,723,724,725,772,773,820,821,868,869,870,871,872]]),
    torch.tensor([[433,527,528,529,530,577,578,579,580,627,674,675,722,723,724,725,772,773,820,821]]),
    torch.tensor([[727,678,677,630,629,628,627,580,579,578,577,576,575,574,573,526,525,524,523,476]]),
    ]
# colors = ['#71A3F7', '#FE6361', '']
fig = plot_lines(amsterdam, metrics_ams, ams_colors, lines=lines, legend_loc="lower left")
fig.show()

fig = plot_lines(amsterdam, metrics_ams, ams_colors, legend_loc="lower left")
fig.show()

#%% Plot Xian lines
lines = [torch.tensor([[287,315,314,313,312,311,310,339,338,367,366,395,394,393,392,391,420,419,448,477,476,505,534,563,562,561,590,589,588,587,616,645,644,643,642,671,670,699,728,786,815,814,813,812]]),
    torch.tensor([[812,756,700,644,587,588,559,530,501,502,503,504,505,476,447,418,389,360,361,332,333,334,305,306,277,278,279,250,221,222,223,224,195,196,197,168,169,170,171,172,173,144,115,86,57]])
    ]
colors = ['#71A3F7', '#FE6361']
fig = plot_lines(xian, metrics_xian, colors, lines=lines)
fig.show()



# %% Amsterdam averages calculation
metrics_ams.index = metrics_ams['model']

ams_empty_ses1 = metrics_ams[ ((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 1)
                            & (metrics_ams['var_lambda'] == 0)]

ams_empty_ses0 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 0)
                            & (metrics_ams['var_lambda'] == 0)]

ams_empty_ggi_2 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'ggi') 
                            & (metrics_ams['ggi_weight'] == 2)]

ams_empty_var_3 = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'group') 
                            & (metrics_ams['var_lambda'] == 3)]

ams_empty_rawls = metrics_ams[((metrics_ams['existing_lines'] == 0) | np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'rawls')]


ams_full_ses1 = metrics_ams[ ((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 1)
                            & (metrics_ams['var_lambda'] == 0)]

ams_full_ses0 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'weighted') 
                            & (metrics_ams['ses_weight'] == 0)
                            & (metrics_ams['var_lambda'] == 0)]

ams_full_var_3 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'group') 
                            & (metrics_ams['var_lambda'] == 3)]

ams_full_ggi_2 = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'ggi') 
                            & (metrics_ams['ggi_weight'] == 2)]

ams_full_rawls = metrics_ams[((metrics_ams['existing_lines'] != 0) & ~np.isnan(metrics_ams['existing_lines']))
                            & (metrics_ams['reward'] == 'rawls')]

#%%
def print_ci(df, col, model: str, z=1.96):
    m = df[col].mean()
    std = df[col].std()
    se = std/math.sqrt(df.shape[0])
    print(f"[{model} - {col}] - Mean: {m} +- {z * se} SE: {se}, CI:({m - z * se}, {m + z * se}), Sample Size: {df.shape[0]}")

def print_stats(df, model:str):
    print_ci(df, 'mean_sat_group_od_pct', model)
    print_ci(df, 'group_pct_gini', model)
    print_ci(df, 'lowest_quintile_sat_od_pct', model)
    print('--------')


print_stats(ams_full_ses0, 'ams_full_ses_0')
print_stats(ams_full_var_3, 'ams_full_var_3')
print_stats(ams_full_rawls, 'ams_full_rawls')
print_stats(ams_full_ggi_2, 'ams_full_ggi_2')

# print_ci(ams_full_ses1, 'mean_sat_group_od_pct', 'ams_full_ses_1')

# print_ci(ams_empty_ses1, 'mean_sat_group_od_pct', 'ams_empty_ses_1')
# print_ci(ams_empty_ses0, 'mean_sat_group_od_pct', 'ams_empty_ses_0')
# print_ci(ams_empty_ggi_2, 'mean_sat_group_od_pct', 'ams_empty_ggi_2')
# print_ci(ams_empty_var_3, 'mean_sat_group_od_pct', 'ams_empty_var_3')

# %%



# ams_empty_ggi_2[['actor_lr', 'critic_lr', 'mean_sat_group_od_pct', 'group_gini', 'budget', 'existing_lines', 'ignore_existing_lines']].sort_values('mean_sat_group_od_pct')