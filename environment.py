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
    mx = torch.zeros((size_x, size_y), device=device)
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
        """Creates a list of tensors for each line, from given grid indices. Used to create line/segment representations of metro lines.

        Args:
            lines (list): list of list of stations (grid indices).

        Returns:
            list: list of tensors, one for each line. Each tensor is a series of vector indices.
        """
        processed_lines = []
        for l in lines:
            l = torch.Tensor(l).long().to(device)
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
        mask_initial = torch.zeros(1, self.grid_size, device=device).long() # 1 : bacth_size
        mask = mask_initial.index_fill_(1, vector_index_allow, 1).float()  # the first 1: dim , the second 1: value

        return mask
    
    # TODO: consider changing this to only calculate the mask based on stationed grids, not every grid covered by old lines.
    def satisfied_od_mask(self, tour_idx: torch.Tensor) -> torch.Tensor:
        """Computes a boolean mask of the satisfied OD flows of a given line (tour).

        Args:
            tour_idx (torch.Tensor): vector indices resembling a line.

        Returns:
            torch.Tensor: mask of self.grid_size * self.grid_size of satisfied OD flows.
        """
        # Satisfied OD pairs from the new line, only considering the new line od demand.
        sat_od_pairs = torch.combinations(tour_idx.flatten(), 2)

        # Satisfied OD pairs from the new line, by considering connections to existing lines.
        # For each line, we look for intersections to the existing lines (full, not only grids with stations).
        # If intersection is found, we add the extra satisfied ODs
        for i, line_full in enumerate(self.existing_lines_full):
            line = self.existing_lines[i]
            intersection_full_line = ((tour_idx - line_full) == 0).nonzero()
            if intersection_full_line.size()[0] != 0:
                intersection_station_line = ((tour_idx - line) == 0).nonzero()

                # We filter the line grids based on the intersection between the new line and the sations of old lines.
                line_mask = torch.ones(line.numel(), dtype=torch.bool)
                line_mask[intersection_station_line[:, 0]] = False
                line_connections = line[line_mask]
                
                # We filter the tour grids based on the intersection between the new line and the full old lines.
                # Note: here we use the full line filter, because we want to leave out the connection of the intersection
                # between the new line and existing line stations, as we assume this is already covered by the existing lines.
                tour_mask = torch.ones(tour_idx.numel(), dtype=torch.bool)
                tour_mask[intersection_full_line[:, 1]] = False
                # Note this won't work with multi-dimensional tour_idx
                tour_connections = tour_idx[0, tour_mask]

                conn_sat_od_pairs = torch.cartesian_prod(tour_connections, line_connections.flatten())
                sat_od_pairs = torch.cat((sat_od_pairs, conn_sat_od_pairs))
        
        # Calculate a mask over the OD matrix, based on the satisfied OD pairs.
        od_mask = torch.zeros(self.grid_size, self.grid_size, device=device).byte()
        od_mask[sat_od_pairs[:, 0], sat_od_pairs[:, 1]] = 1
        
        return od_mask

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

        # Build the normalized OD and SES matrices.
        self.od_mx = matrix_from_file(path / 'od.txt', self.grid_size, self.grid_size)
        # TODO: consider doing the normalization in matrix_from_file with an argument.
        self.od_mx = self.od_mx / torch.max(self.od_mx)
        try:
            self.price_mx = matrix_from_file(path / 'average_house_price_gid.txt', self.grid_x_size, self.grid_y_size)
            self.price_mx = self.price_mx / torch.max(self.price_mx)
        except FileNotFoundError:
            print('Price matrix not available.')

        # Read existing metro lines of the environment.
        # json is used to load lists from ConfigParser as there is no built in way to do it.
        existing_lines = self.process_lines(json.loads(config.get('config', 'existing_lines')))
        # Full lines contains the lines + the squares between consecutive stations e.g. if line is (0,0)-(0,2)-(2,2) then full line also includes (0,1), (1,2).
        # These are important for when calculating connections between generated & lines and existing lines.
        existing_lines_full = self.process_lines(json.loads(config.get('config', 'existing_lines_full')))

        # Create line tensors
        self.existing_lines = [l.view(len(l), 1) for l in existing_lines]
        self.existing_lines_full = [l.view(len(l), 1) for l in existing_lines_full]

        # Apply excluded OD segments to the od_mx. E.g. segments very close to the current lines that we want to set OD to 0.
        if config.has_option('config', 'excluded_od_segments'):
            exclude_segments = self.process_lines(json.loads(config.get('config', 'excluded_od_segments')))
            if len(exclude_segments) > 0:
                exclude_pairs = torch.Tensor([]).long().to(device)
                for s in exclude_segments:
                    # Create two-way combinations of each segment.
                    # e.g. segment: 1-2-3-4, pairs: 1-2, 2-1, 1-3, 3-1, 1-4, 4-1, ... etc
                    pair1 = torch.combinations(s, 2)
                    pair2 = torch.combinations(s.flip(0), 2)

                    exclude_pairs = torch.cat((exclude_pairs, pair1, pair2))
            
                self.od_mx[exclude_pairs[:, 0], exclude_pairs[:, 1]] = 0

        # Create the static representation of the grid coordinates - to be used by the actor.
        xs, ys = [], []
        for i in range(self.grid_x_size):
            for j in range(self.grid_y_size):
                xs.append(i)
                ys.append(j)
        self.static = torch.Tensor([[xs, ys]]).to(device) # should be float32
