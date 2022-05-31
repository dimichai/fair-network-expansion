#%%
import itertools
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
ams_nb = gpd.read_file('./ams-districts.geojson')
# ams_nb = gpd.read_file('./ams-neighbourhoods.geojson')
# For population/income data by buurt/wijk
ams_ses = pd.read_csv('./ams-ds-ses.csv')
ams_ses = ams_ses[['WK_CODE', 'pop', 'avg_inc_per_res']]

# ams_nb = ams_nb.to_crs(epsg=3035)
# %%
xmin, ymin, xmax, ymax = ams_nb.total_bounds

n = 45
cell_size = (xmax-xmin)/n

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

# grid = gpd.GeoDataFrame({'geometry':polygons}, crs='EPSG:4326')
grid = gpd.GeoDataFrame(grid)
# grid = grid.to_crs('EPSG:4326')
# grid = grid.to_crs('EPSG:3035')
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
                metro_stops['LAT']))

# metro_stops = metro_stops.to_crs('EPSG:4326')
# metro_stops = metro_stops.to_crs(epsg=3035)

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

# some tests for avg income
# wm = lambda x: np.average(x, weights=overlay_pct.loc[overlay_pct.index, 'grid_pop'], axis=0)
# overlay_pct.groupby('v').agg(grid_pop=('grid_pop', 'sum'), avg_income=('avg_inc_per_res', wm))

# Merge gridpop with the original grid
grid = grid.merge(gridpop, on='v', how='left')
grid['grid_pop'] = grid['grid_pop'].fillna(0)
grid.to_csv('./amsterdam_grid.csv', index=False)

#%% Plot the Grid environment and the existing lines.
fig, ax = plt.subplots(figsize=(15, 10))
gridenv = np.zeros((grid_x_size, grid_y_size))
for i, row in grid.iterrows():
    gridenv[row['g_x'], row['g_y']] = row['grid_pop']
# gridenv = np.random.rand(grid_x_size, grid_y_size)
        # gridenv[row['g_x'], row['g_y']] = int(metro_labels[i])

ax.imshow(gridenv, cmap='Blues')
markers = itertools.cycle(['o','s','v', '^', 'P', 'h'])
for i, l in enumerate(metro_lines):
    l_v = l.sjoin(grid)
    l_v = l_v.sort_values('v')

    ax.plot(l_v['g_y'], l_v['g_x'], 'o', marker=next(markers), label=metro_labels[i], markersize=10, alpha=0.5)

ax.legend()


# %% Print labels of the Grid
fig, ax = plt.subplots(figsize=(15, 10))
grid['coords'] = grid['geometry'].apply(lambda x: x.representative_point().coords[:])
grid['coords'] = [coords[0] for coords in grid['coords']]

grid.boundary.plot(ax=ax, edgecolor='gray')
for idx, row in grid.iterrows():
    plt.annotate(text=row['v'], xy=row['coords'],
                 horizontalalignment='center', fontsize=8)
# %%
