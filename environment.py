import configparser
import torch
import json
import numpy as np
import constants

device = constants.device

def matrix_from_file(path, size_x, size_y):
    """Reads a grid file that is structured as idx1,idx2,weight and creates a matrix representation of it.
    It is used to read ses variables per grid, od matrices, etc.
    Note: non-existing values are assigned 0.

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
        v_idx = grid_idx[:, 0] * self.grid_y_size + grid_idx[:, 1]
        return v_idx

    def vector_to_grid(self, vector_idx):
        """Converts vector index (x^2) to grid indices (x, y)

        Args:
            vector_idx (torch.Tensor): the vector index to be converted to grid indices: x

        Returns:
            torch.Tensor: covnerted grid index.
        """

        grid_x = (vector_idx // self.grid_y_size)
        grid_y = (vector_idx % self.grid_y_size)

        # Control for when vector_idx is just a tensor of 1 idx vs when it is a tensor of multiple idxs.
        # TODO: maybe there is a way to do that universally without if
        if vector_idx.dim() == 0:
            grid_x = grid_x.view(1)
            grid_y = grid_y.view(1)

            return torch.cat((grid_x, grid_y), dim=0).view(1, 2)

        # grid_x = grid_x.unsqueeze(dim=0)
        # grid_y = grid_y.unsqueeze(dim=0)
        return torch.cat((grid_x, grid_y), dim=0)#.transpose(-1, -2)
        
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
    

    def matrix_reward_scaling(self, tour_idx: torch.Tensor):
        tour_idx = tour_idx[0].flatten()
        tour_dict = {(s1, s2): tour_idx[i:j+1] for i, s1 in enumerate(tour_idx) for j, s2 in enumerate(tour_idx) if i != j and i < j}
        od_scaling_mask = torch.zeros(self.grid_size, self.grid_size, device=device)

        for c, subtour in tour_dict.items():
            # Calculate direct euclidian distance
            p1 = self.vector_to_grid(c[0]).float()
            p2 = self.vector_to_grid(c[1]).float()
            d_eucl = torch.dist(p1, p2, p=2)

            # Calculate distance via subtour
            points = [self.vector_to_grid(p).float() for p in subtour]
            d_tour = sum([torch.dist(points[i], points[i+1]) for i, _ in enumerate(points) if i != len(points) - 1])

            # Calculate scaling factor for subtour connection
            scaled = self.reward_scaling_fn(d_eucl, d_tour)
            od_scaling_mask[c[0], c[1]] = scaled

        return od_scaling_mask

    def get_groups_file(self, environment):
        if environment == "diagonal_5x5" or environment == "dilemma_5x5":
            return "groups.txt"
        elif environment == "amsterdam" or environment == "xian":
            return "price_groups_5.txt"
        else:
            return None

    def __init__(self, env_path, groups_file=None, reward_scaling_fn=None):
        """Initialise city environment.

        Args:
            env_path (Path): path to the folder that contains the needed initialisation files of the environment.
            groups_file (str): file within envirnoment folder that contains group membership for each grid square.
        """
        super(Environment, self).__init__()

        if not groups_file:
            groups_file = self.get_groups_file(str(env_path).split("/")[-1])
            print("No groups file provided. Trying to use the default groups file.")

        # read configuration file that contains basic parameters for the environment.
        config = configparser.ConfigParser()
        config.read(env_path / 'config.txt')
        assert 'config' in config, f"No config found! Please check if the environment {env_path} exists"

        # size of the grid
        self.grid_x_size = config.getint('config', 'grid_x_size')
        self.grid_y_size = config.getint('config', 'grid_y_size')
        self.grid_size = self.grid_x_size * self.grid_y_size

        # size of the model's static and dynamic parts
        # self.static_size = config.getint('config', 'static_size')
        # self.dynamic_size = config.getint('config', 'dynamic_size')

        # Build the normalized OD and SES matrices.
        self.od_mx = matrix_from_file(env_path / 'od.txt', self.grid_size, self.grid_size)
        self.od_mx = self.od_mx / torch.max(self.od_mx)
        try:
            self.price_mx = matrix_from_file(env_path / 'average_house_price_gid.txt', self.grid_x_size, self.grid_y_size)
            self.price_mx_norm = self.price_mx / torch.max(self.price_mx)
        except FileNotFoundError:
            print('Price matrix not available.')
        
        # If there are group memberships of each grid square, then create an OD matrix for each group.
        self.group_od_mx = None # initialize it so we can check later on if it has any value
        if groups_file:
            self.grid_groups = matrix_from_file(env_path / groups_file, self.grid_x_size, self.grid_y_size)
            # matrix_from_file initializes a tensor with torch.zeros - we convert them to nans
            self.grid_groups[self.grid_groups == 0] = float('nan')
            # Get all unique groups
            groups = self.grid_groups[~self.grid_groups.isnan()].unique()
            # Create a group-specific od matrix for each group.
            self.group_od_mx = []
            for g in groups:
                group_mask = torch.zeros(self.od_mx.shape, device=device)
                group_squares = self.grid_to_vector((self.grid_groups == g).nonzero())
                # Original OD matrix is symmetrical, so group OD matrices should also be symmetrical.
                group_mask[group_squares, :] = 1
                group_mask[:, group_squares] = 1
                self.group_od_mx.append(group_mask * self.od_mx)

        # Read existing metro lines of the environment.
        if config.has_option('config', 'existing_lines'):
            # json is used to load lists from ConfigParser as there is no built in way to do it.
            existing_lines = self.process_lines(json.loads(config.get('config', 'existing_lines')))
            # Full lines contains the lines + the squares between consecutive stations e.g. if line is (0,0)-(0,2)-(2,2) then full line also includes (0,1), (1,2).
            # These are important for when calculating connections between generated & lines and existing lines.
            existing_lines_full = self.process_lines(json.loads(config.get('config', 'existing_lines_full')))

            # Create line tensors
            self.existing_lines = [l.view(len(l), 1) for l in existing_lines]
            self.existing_lines_full = [l.view(len(l), 1) for l in existing_lines_full]
        else:
            self.existing_lines = []
            self.existing_lines_full = []

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

        # Set the right scaling function for renewed reward function
        if reward_scaling_fn == "inverse":
            self.reward_scaling_fn = lambda d_eucl, d_tour: torch.nn.functional.relu(d_eucl/d_tour)
        elif reward_scaling_fn == "linear":
            self.reward_scaling_fn = lambda d_eucl, d_tour: torch.nn.functional.relu(1 - ((d_tour-1)/d_eucl))

        # Create the static representation of the grid coordinates - to be used by the actor.
        xs, ys = [], []
        for i in range(self.grid_x_size):
            for j in range(self.grid_y_size):
                xs.append(i)
                ys.append(j)
        self.static = torch.Tensor([[xs, ys]]).to(device) # should be float32
