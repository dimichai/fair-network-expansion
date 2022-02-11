import configparser
import torch

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def matrix_from_file(path, size_x, size_y):
    """Reads a grid file that is structured as idx1,idx2,weight and creates a matrix representation of it.
    It is used to read ses variables per grid, od matrices, etc.

    Args:
        path (string): the path of the file.
        size_x (int): nr of rows in the matrix.
        size_y (int): nr of columns in the matrix.

    Returns:
        torch.Tensor: the matrix representation of the file.
    """
    mx = torch.zeros((size_x, size_y)).to(device)
    with open(path, 'r') as f:
        for line in f:
            idx1, idx2, weight = line.rstrip().split(',')
            idx1 = int(idx1)
            idx2 = int(idx2)
            weight = float(weight)

            mx[idx1][idx2] = weight
    return mx

class Environment(object):
    """The City Grid environment the agent learns from."""

    def __init__(self, path):
        """Initialise city environment.

        Args:
            path (Path): path to the folder that contains the needed initialisation files of the environment.
        """
        super(Environment, self).__init__()

        # read configuration file that contains basic parameters for the environment.
        config = configparser.ConfigParser()
        config.read(path / 'config.txt')
        assert 'config' in config

        # size of the grid
        self.grid_x_size = config.getint('config', 'grid_x_size')
        self.grid_y_size = config.getint('config', 'grid_y_size')
        self.grid_size = self.grid_x_size * self.grid_y_size
        
        # Build a grid_size x grid_size OD matrix.
        self.od_mx = matrix_from_file(path / 'od.txt', self.grid_size, self.grid_size)
        self.price_mx = matrix_from_file(path / 'average_house_price_gid.txt', self.grid_x_size, self.grid_y_size)
        
