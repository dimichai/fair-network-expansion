import configparser
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        # Build a grid_x_size * grid_y_size OD matrix from the od.txt file
        self.od_mx = torch.zeros((self.grid_size, self.grid_size)).to(device)
        with open(path / 'od.txt', 'r') as f:
            for line in f:
                index1, index2, weight = line.rstrip().split(',')
                index1 = int(index1)
                index2 = int(index2)
                weight = float(weight)

                self.od_mx[index1][index2] = weight


