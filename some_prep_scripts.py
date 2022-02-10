#%%
import numpy as np

od_mx = np.zeros((29*29, 29*29))

def g_to_v(gx, gy, grid_x_max):
    index = int(gx) * grid_x_max + int(gy)

    return index


data = []
with open('./environments/xian/od.txt', 'r') as f:
    for line in f:
        index1, index2, weight = line.rstrip().split('\t')
        index1 = int(index1)
        index2 = int(index2)
        weight = float(weight)

        data.append((index1, index2, weight))
# %%
with open('./environments/xian/od_new.txt', 'w+') as f:
    for d in data:
        f.write(f'{d[0]},{d[1]},{d[2]}\n')

# %%


f = open('./environments/xian/average_house_price.txt', 'r')

prices = []
for line in f:
    grid,price = line.rstrip().split('\t')
    index_x, index_y = grid.split(',')

    index_x = int(index_x)
    index_y = int(index_y)
    price = float(price)

    idx = g_to_v(index_x, index_y, 29)

    prices.append((idx, price))
f.close()

with open('./environments/xian/average_house_orice_idx.txt', 'w+') as f:
    for p in prices:
        f.write(f'{p[0]},{p[1]}\n')

# %%
