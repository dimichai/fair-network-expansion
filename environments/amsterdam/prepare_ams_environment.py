#%%
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
ams_nb = gpd.read_file('./ams-neighbourhoods.geojson')
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

# Metro Lines
# metro_lines = pd.read_csv('./metro_lines.csv')
# metro_lines = gpd.GeoDataFrame(metro_lines, geometry=gpd.points_from_xy(metro_lines['x'], metro_lines['y']))

metro_stops = pd.read_csv('./TRAMMETRO_PUNTEN_2022.csv', delimiter=';')
metro_stops = metro_stops[metro_stops['Modaliteit'] == 'Metro']
metro_stops = gpd.GeoDataFrame(metro_stops, geometry=gpd.points_from_xy(metro_stops['LNG'], metro_stops['LAT']))

fig, ax = plt.subplots(figsize=(15, 10))
ams_nb.plot(ax=ax)
grid.boundary.plot(ax=ax, edgecolor='gray')
metro_stops.plot(ax=ax, color='orange', markersize=40)
fig.suptitle(f'Amsterdam Environment Grid - Total Cells: {grid.shape[0]}', fontsize=25)
fig.tight_layout()
# ax.set_axis_off()
fig.savefig(f'./amsterdam_env_{len(rows)}x{len(cols)}.png')

# Create metro lines
metro_labels = ['50', '51', '52', '53', '54']
metro_lines = []
for label in metro_labels:
    metro_lines.append(metro_stops[metro_stops['Lijn_select'].str.contains('50')])

grid.to_csv('./amsterdam_grid.csv', index=False)
# %%
# Print labels
# fig, ax = plt.subplots(figsize=(15, 10))
# grid['coords'] = grid['geometry'].apply(lambda x: x.representative_point().coords[:])
# grid['coords'] = [coords[0] for coords in grid['coords']]

# grid.boundary.plot(ax=ax, edgecolor='gray')
# for idx, row in grid.iterrows():
#     plt.annotate(text=row['v'], xy=row['coords'],
#                  horizontalalignment='center', fontsize=8)
# %%
