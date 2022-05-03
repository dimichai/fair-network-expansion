import numpy as np
# from https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python 
# but removed weights
def gini(x):
    # The rest of the code requires numpy arrays.
    # x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
