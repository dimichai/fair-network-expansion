import configparser
import torch
import json
import numpy as np
import constants

device = constants.device

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

    def grid_to_vector(self, grid_idx):
        """Converts grid indices(x, y) to a vector index (x^).

        Args:
            grid_idx (torch.Tensor): the grid indices to be converted to vector indices: [[x1, y1], [x2, y2], ...]

        Returns:
            torch.Tensor: Converted vector.
        """
        v_idx = grid_idx[:, 0] * self.grid_x_size + grid_idx[:, 1]
        return v_idx

    def vector_to_grid(self, vector_idx):
        """Converts vector index (x^2) to grid indices (x, y)

        Args:
            vector_idx (torch.Tensor): the vector index to be converted to grid indices: x

        Returns:
            torch.Tensor: covnerted grid index.
        """

        grid_x = (vector_idx // self.grid_x_size).view(1)
        grid_y = (vector_idx % self.grid_y_size).view(1)

        return torch.cat((grid_x, grid_y), dim=0).view(1, 2)

    def process_lines(self, lines):
        processed_lines = []
        for l in lines:
            l = torch.Tensor(l).to(device)
            # Convert grid indices (x,y) to vector indices (x^)
            l = self.grid_to_vector(l)
            processed_lines.append(l)
        return processed_lines

    def update_mask(self, vector_index_allow):
        """Updates the selection mask. Only allowed next locations are assigned 1, all others 0.
        This prevents re-selecting locations.

        Args:
            vector_index_allow (torch.Tensor): Allowed locations(indices) to be selected.

        Returns:
            torch.Tensor: the updated mask of allowed next locations.
        """
        mask_initial = torch.zeros(1, self.grid_num, device=device).long() # 1 : bacth_size
        mask = mask_initial.index_fill_(1, vector_index_allow, 1).float()  # the first 1: dim , the second 1: value

        return mask


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

        # size of the model's static and dynamic parts
        # self.static_size = config.getint('config', 'static_size')
        # self.dynamic_size = config.getint('config', 'dynamic_size')

        # Build the OD and the SES matrices.
        self.od_mx = matrix_from_file(path / 'od.txt', self.grid_size, self.grid_size)
        self.price_mx = matrix_from_file(path / 'average_house_price_gid.txt', self.grid_x_size, self.grid_y_size)

        # Read existing metro lines of the environment.
        # json is used to load lists from ConfigParser as there is no built in way to do it.
        existing_lines = self.process_lines(json.loads(config.get('config', 'existing_lines')))
        # Full lines contains the lines + the squares between consecutive stations e.g. if line is (0,0)-(0,2)-(2,2) then full line also includes (0,1), (1,2).
        # These are important for when calculating connections between generated & lines and existing lines.
        existing_lines_full = self.process_lines(json.loads(config.get('config', 'existing_lines_full')))

        # Create line tensors
        self.existing_lines = [l.view(len(l), 1) for l in existing_lines]
        self.existing_lines_full = [l.view(len(l), 1) for l in existing_lines_full]
        