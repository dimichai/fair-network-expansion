#%%
import itertools
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
ams_nb = gpd.read_file('./ams-districts.geojson', crs='EPSG:4326')
# ams_nb = gpd.read_file('./ams-neighbourhoods.geojson')
# For population/income data by buurt/wijk
ams_ses = pd.read_csv('./ams-ds-ses.csv')
ams_ses = ams_ses[['WK_CODE', 'pop', 'avg_inc_per_res']]


# Amersfoort / RD New -- Netherlands - Holland - Dutch
CRS = 'EPSG:28992'
# Flatten the neighborhoods
ams_nb = ams_nb.to_crs(CRS)
# %%
xmin, ymin, xmax, ymax = ams_nb.total_bounds

# n = 45
# cell_size = (xmax-xmin)/n
cell_size = 500

cols = list(np.arange(xmin, xmax + cell_size, cell_size))
rows = list(np.arange(ymin, ymax + cell_size, cell_size))

grid = {'geometry': [], 'lat': [], 'lon': [], 'g_x': [], 'g_y': [], 'v': []}
# total cell counter/index
v = 0 
# We reverse the rows here because we want the index of the cells to start from top to bottom.
for i, y in enumerate(reversed(rows[:-1])):
    for j, x in enumerate(cols[:-1]):
        grid['lat'].append(y)
        grid['lon'].append(x)
        grid['g_x'].append(i)
        grid['g_y'].append(j)
        grid['v'].append(v)
        v += 1
        grid['geometry'].append(Polygon([(x,y), (x+cell_size, y), (x+cell_size, y+cell_size), (x, y + cell_size)]))

grid = gpd.GeoDataFrame(grid, crs=CRS)
grid_x_size = grid.g_x.max() + 1
grid_y_size = grid.g_y.max() + 1

# Metro Lines
# metro_lines = pd.read_csv('./metro_lines.csv')
# metro_lines = gpd.GeoDataFrame(metro_lines, geometry=gpd.points_from_xy(metro_lines['x'], metro_lines['y']))

metro_stops = pd.read_csv('./TRAMMETRO_PUNTEN_2022.csv', delimiter=';')
metro_stops = metro_stops[metro_stops['Modaliteit'] == 'Metro']
metro_stops = gpd.GeoDataFrame(
                metro_stops, 
                geometry=gpd.points_from_xy(metro_stops['LNG'], 
                metro_stops['LAT']),
                crs='EPSG:4326')
metro_stops = metro_stops.to_crs(CRS)

fig, ax = plt.subplots(figsize=(15, 10))
ams_nb.plot(ax=ax)
grid.boundary.plot(ax=ax, edgecolor='gray')
metro_stops.plot(ax=ax, color='orange', markersize=40)
fig.suptitle(f'Amsterdam Environment Grid - Total Cells: {grid.shape[0]}', fontsize=25)
fig.tight_layout()
# ax.set_axis_off()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_wijk.png')

# Create metro lines
metro_labels = ['50', '51', '52', '53', '54']
metro_lines = []
for label in metro_labels:
    metro_lines.append(metro_stops[metro_stops['Lijn_select'].str.contains(label)])

#%% Overlay the grid over the neighborhoods to calculate population by grid
# https://gis.stackexchange.com/questions/421888/getting-the-percentage-of-how-much-areas-intersects-with-another-using-geopandas

grid['area_grid'] = grid.area
if CRS == 'EPSG:28992':
    grid['area_grid_km'] = grid['area_grid'] / 10**6
ams_nb['area_nb'] = ams_nb.area

overlay = grid.overlay(ams_nb, how='intersection')
overlay['area_joined'] = overlay.area
overlay['area_overlay_pct'] = overlay['area_joined'] / overlay['area_nb']

overlay_pct = (overlay
           .groupby(['v','WK_CODE'])
           .agg({'area_overlay_pct':'sum'}))

# Plot the distribution of covered neighborhoods per grid
counts, edges, bars = plt.hist(
    overlay_pct.value_counts('v').values, 
    weights=np.ones(len(overlay_pct.value_counts('v').values)) / len(overlay_pct.value_counts('v').values))
counts = counts.round(2)
plt.bar_label(bars, labels=counts)
plt.title('Distribution of covered neighborhoods per square grid')
plt.savefig('./grid_to_nb_distribution.png')

# Assign population to each grid.
overlay_pct = overlay_pct.reset_index().merge(ams_ses, on='WK_CODE', how='left')
overlay_pct['grid_pop'] = overlay_pct['area_overlay_pct'] * overlay_pct['pop']
overlay_pct['grid_pop'] = overlay_pct['grid_pop'].round()
gridpop = overlay_pct.groupby('v')[['grid_pop']].sum().reset_index()

# Assign average income to each grid.
# https://stackoverflow.com/questions/31521027/groupby-weighted-average-and-sum-in-pandas-dataframe
def weighted_average(df, data_col, weight_col, by_col):
    df['_data_times_weight'] = df[data_col] * df[weight_col]
    df['_weight_where_notnull'] = df[weight_col] * pd.notnull(df[data_col])
    g = df.groupby(by_col)
    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
    del df['_data_times_weight'], df['_weight_where_notnull']
    return result

gridinc = weighted_average(overlay_pct, 'avg_inc_per_res', 'area_overlay_pct', 'v').to_frame('grid_avg_inc')

# Merge gridpop with the original grid
grid = grid.merge(gridpop, on='v', how='left')
grid = grid.merge(gridinc, on='v', how='left')
# Population Density per square
grid['pop_density_km'] = grid['grid_pop'] / grid['area_grid_km']
# grid['grid_pop'] = grid['grid_pop'].fillna(0)
grid.to_csv('./amsterdam_grid.csv', index=False)

#%% Plot the Grid environment and the existing lines.
gridenv = np.zeros((grid_x_size, grid_y_size))
for i, row in grid.iterrows():
    gridenv[row['g_x'], row['g_y']] = row['grid_pop']

gridenvinc = np.zeros((grid_x_size, grid_y_size))
for i, row in grid.iterrows():
    gridenvinc[row['g_x'], row['g_y']] = row['grid_avg_inc']

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(gridenv, cmap='Blues')
markers = itertools.cycle(['o','s','v', '^', 'P', 'h'])
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_v = l_v.sort_values('v')

    ax.plot(l_v['g_y'], l_v['g_x'], 'o', marker=next(markers), label=metro_labels[i], markersize=10, alpha=0.5)

fig.suptitle('Amsterdam Grid Population - Existing Lines', fontsize=30)
ax.legend()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_population.png')

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(gridenvinc, cmap='Blues')
markers = itertools.cycle(['o','s','v', '^', 'P', 'h'])
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_v = l_v.sort_values('v')

    ax.plot(l_v['g_y'], l_v['g_x'], 'o', marker=next(markers), label=metro_labels[i], markersize=10, alpha=0.5)

fig.suptitle('Amsterdam Grid Avg Income - Existing Lines', fontsize=30)
ax.legend()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_avg_income.png')

# %% Print labels of the Grid
fig, ax = plt.subplots(figsize=(15, 10))
grid['coords'] = grid['geometry'].apply(lambda x: x.representative_point().coords[:])
grid['coords'] = [coords[0] for coords in grid['coords']]

grid.boundary.plot(ax=ax, edgecolor='gray')
for idx, row in grid.iterrows():
    plt.annotate(text=row['v'], xy=row['coords'],
                 horizontalalignment='center', fontsize=8)
# %% Calculate OD Flows using the Universal Mobility Law
d = 7
fmin = 1/d
fmax = d
od_mx = np.zeros((grid.shape[0], grid.shape[0]))
for i,row_i in grid.iterrows():
    for j, row_j in grid.iterrows():
        if i == j:
            continue
        
        # destination attractiveness
        mu_j = row_j['pop_density_km'] * row_j['area_grid_km'] ** 2 * fmax
        
        if np.isnan(mu_j):
            mu_j = 0
        
        # Manhattan distance
        r_ij = abs(row_i['g_x'] - row_j['g_x']) + abs(row_i['g_y'] - row_j['g_y'])
        if np.isnan(r_ij):
            print(f'Distance between {i} and {j} is nan - this is a bug and should not happen')
        # Origin Destination flow estimate
        od_ij = mu_j * row_i['area_grid_km'] / r_ij ** 2 * np.log(fmax/fmin)
        od_mx[i, j] = od_ij


# %% Plot aggregate OD flow for each grid cell
agg_od_g = np.zeros((grid_x_size, grid_y_size))
agg_od_v = od_mx.sum(axis=1)
# Get the grid indices.
for i in range(agg_od_v.shape[0]):
    agg_od_g[grid.iloc[i]['g_x'], grid.iloc[i]['g_y']] = agg_od_v[i]

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(agg_od_g, cmap='Blues')
fig.suptitle('Amsterdam Agregate OD flows', fontsize=30)
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_agg_od_flows.png')

# %% Plot correlation between population and aggregate d
fig, ax = plt.subplots(figsize=(7, 5))
grid['aggregate_od'] = agg_od_v
corr = grid[['grid_pop', 'aggregate_od']].corr().iloc[0, 1]
grid.plot.scatter('grid_pop', 'aggregate_od', ax=ax)
fig.suptitle(f'Aggregate OD flows vs Population: Pearson: {round(corr, 3)}')
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}_agg_od_vs_pop.png')
# %%