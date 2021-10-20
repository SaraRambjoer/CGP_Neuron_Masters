# here define the world space


class Neuron():
    def __init__(self, grid, x, y, z) -> None:
        self.grid = grid
        self.x = x
        self.y = y
        self.z = z
    
    def set_grid(self, grid):
        self.grid = grid

class NeuronEngine():
    def __init__(self, input_arity, output_arity, grid_count, grid_size, actions_max, debugging = False):
        self.input_arity = input_arity
        self.output_arity = output_arity
        self.grid_count = grid_count  # Grids in grid grid per dimension
        self.grid_size = grid_size   # neurons per grid per dimension
        self.actions_max = actions_max
        self.actions_count = 0
        self.input_neurons = []
        for _ in range(input_arity):
            self.input_neurons.append(InputNeuron())
        self.output_neurons = []
        for _ in range(output_arity):
            self.output_neurons.append(OutputNeuron())
        self.init_grids()
        self.init_neurons()
        self.action_queue = []
        self.timestep_indexes = []
        self.debugging = debugging
        self.timestep = 0  # TODO Add tracking which timestep we are on
    
    def get_current_timestep(self):
        return self.timestep
    
    # RFE assumes knowledge of neuron class properties
    def update_neuron_position(self, neuron, neuron_pos_loc):
        neuron_grid = neuron.grid
        (xg, yg, zg) = (neuron_grid.x, neuron_grid.y, neuron_grid.z)
        (xl, yl, zl) = neuron_pos_loc
        new_xg, new_yg, new_zg = (0, 0, 0)
        new_loc_x, new_loc_y, new_loc_g = (0, 0, 0)
        new_glob_x, new_glob_y, new_glob_z = (0, 0, 0)
        # TODO fix this, finish pattern and use it in each dimension
        def _modify_pattern(local_pos, grid_size, global_pos, grid_pos, grid_count):
            if local_pos > grid_size:
                if grid_pos < grid_count:
                    new_local = 0
                    new_grid = grid_pos + 1
                    new_global = global_pos + 1
                else: 
                    new_local = local_pos - 1
                    new_grid = grid_pos
                    new_global = global_pos
            elif local_pos < 0:
                if grid_pos > 0:
                    new_local = grid_size
                    new_grid = grid_pos - 1
                    new_global = global_pos - 1
                else: 
                    new_local = local_pos + 1
                    new_glob = global_pos
                    new_grid = grid_pos
            else: 
                new_grid = grid_pos
                new_local = local_pos
                new_global = ??? # TODO global pos is fucked because it needs to be set when moving in a direction
                # or info about direction needs to be transmitted, as it is missing info for this step. 


        # RFE horrible knowledge duplication
        if xl > self.grid_size:
            if xg < self.grid_size:
                new_loc_x = 0
                new_xg = xg + 1
                new_glob_x = neuron.x_glob + 1
            else:
                new_loc_x = xl - 1
                new_xg = xg
                new_glob_x = neuron.x_glob
        elif xl < 0:
            if xg > 0:
                new_loc_x = self.grid_size
                new_xg = xg - 1
                new_glob_x = neuron.x_glob - 1
            else: 
                new_loc_x = 0
                new_glob_x = neuron.x_glob
        if yl > self.grid_size:
            if yg < self.grid_size:
                new_loc_y = 0
                new_yg = yg + 1
                new_glob_x = neuron.y_glob + 1
            else:
                new_yg = yg
                new_loc_y = self.grid_size
                new_glob_y = neuron.y_glob
        elif yl < 0:
            if yg > 0:
                new_loc_y = self.grid_size
                new_yg = yg - 1
                new_glob_y = neuron.y_glob - 1
            else:
                new_loc_y = 0
                new_yg = yg
                new_glob_y = neuron.y_glob
        if xl > self.grid_size:
            if xg < self.grid_size:
                new_loc_z = 0
                new_zg = zg + 1
                new_glob_z = neuron.z_glob + 1
            else:
                new_zg = zg
                new_loc_z = self.grid_size
                new_glob_z = neuron.z_glob
        elif zl < 0:
            if zl > 0:
                new_loc_z = self.grid_size
                new_zg = zg - 1
                new_glob_z = neuron.z_glob - 1
            else:
                new_loc_z = 0
                new_zg = zg
                new_glob_z = neuron.z_glob + 1
        


        # return new grid and new local position in grid and new global position (in case of edges)

    def init_grids(self):
        self.grids = {}
        for n1 in range(self.grid_count):
            for n2 in range(self.grid_count):
                for n3 in range(self.grid_count):
                    self.grids["".join(str(n1), str(n2), str(n3))] = Grid(n1, n2, n3, self.grid_count)
    
    def init_neurons(self):
        pass
        # Should set up input and output neurons properly, as well as the one starting neuron. 
        
    def reset(self):
        self.actions_count = 0
        self.init_grids()
        self.init_neurons()

    def approximate_distance_neuron(self, neuron1, neuron2):
        return self.approximate_distance_grid(neuron1.grid, neuron2.grid)

    def approximate_distance_neuron_gridsearch(self, neuron1, neuron2):
        # Should not be used as neurons SHOULD know which grid they are in order to avoid grid search
        grid1 = None
        grid2 = None
        for grid in self.grids.values():
            if grid.contains(neuron1):
                if grid1 is not None:
                    raise Exception("Neuron found in several grids")
                grid1 = grid
            if grid.contains(neuron2):
                if grid2 is not None: 
                    raise Exception("Neuron found in several grids")
                grid2 = grid
        return self.approximate_distance_grid(grid1, grid2)
    
    def approximate_distance_grid(self, grid1, grid2):
        # If slow consider manhattan distance
        # to be used to determine candidate neurons of a given distance, as gridwise distance helps limit search space from all neurons to only ones in 
        # approximate distance grids. 
        return ((grid1.x-grid2.x)**2 + (grid2.y-grid2.y)**2 + (grid1.z - grid2.z)**2)*self.grid_size
    
    def correct_neuron_distance(self, neuron1, neuron2):
        x0, y0, z0 = neuron1.grid.to_global((neuron1.x, neuron1.y, neuron1.z))
        x1, y1, z1 = neuron2.grid.to_global((neuron2.x, neuron2.y, neuron2.z))
        return (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2
    
    def run_sample(self, input_sample):
        # TODO set input neurons to values in input sample
        # TODO setup action queue.
        # Consists of elements (timestep, action_func)
        self.action_queue = []
        while len(self.action_queue > 0) and self.actions_count < self.actions_max:
            _, action = self.action_queue.pop(0)
            action(self)  # runs action
        # TODO get output from output neurnos, return it

    def run(self, inputs, eval_func):
        # for every sample in inputs run it, some way of evaluating result as well
        pass

    def add_action_to_queue(self, action, timestep):
        # TODO update such that it functions with id's and removing them on death
        timesteps = [x[0] for x in self.timestep_indexes]
        timestep_relative_order = 0
        for x in timesteps:
            if timestep > x: 
                timestep_relative_order += 1
        if timestep_relative_order > len(self.timestep_indexes):
            self.action_queue.append((timestep, action))
            self.timestep_indexes.append((timestep, len(self.action_queue-1)))
        else:
            pos = self.timestep_indexes[timestep_relative_order][1]+1
            self.action_queue.insert(pos, (timestep, action))
            self.timestep_indexes.insert(timestep_relative_order, (timestep, pos))
            for x in self.timestep_indexes[timestep_relative_order+1:]:
                x[1] = x[1]+1

    def remove_neuron(self, neuron):
        pass
        # TODO handle removing neuron with id from queue on death, along with their dendrites and axons, while
        # should also handle breaking connections
        # should use a helepr function for removing form queue whihc also works with axons and dendrites

    def add_neuron(self, neuron_pos, neuron_internal_state):
        pass
        # TODO, called on neuron birth. Add to grid and other ways fo tracking existing neurons. 
        # Returns copy of neuron

class Grid():
    def __init__(self, x, y, z, size) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.neurons = []
    
    def contains(self, neuron):
        return neuron.id in [x.id for x in self.neurons]
    
    def to_global(self, pos):
        # pos = (x, y, z)
        g_x = pos[0] + self.x*self.size
        g_y = pos[1] + self.y*self.size
        g_z = pos[2] + self.z*self.size
        return (g_x, g_y, g_z)
               


class InputNeuron(Neuron):
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in
        super(Neuron, self).__init__(grid)
        self.x = x
        self.y = y
        self.z = z

class OutputNeuron(Neuron):
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in 
        super(Neuron, self).__init__(grid)
        self.x = x
        self.y = y
        self.z = z