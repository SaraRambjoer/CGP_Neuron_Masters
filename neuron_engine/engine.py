# here define the world space
import random
from neuron import Neuron, Axon
import math

class NeuronEngine():
    def __init__(self,
                 input_arity, 
                 output_arity, 
                 grid_count, 
                 grid_size, 
                 actions_max, 
                 neuron_initialization_data,
                 axon_initialization_data,
                 signal_arity,
                 hox_variant_count,
                 counter,
                 instances_per_iteration,
                 debugging = False):

        self.instances_per_iteration = instances_per_iteration
        self.neuron_initialization_data = neuron_initialization_data
        self.axon_initialization_data = axon_initialization_data
        self.signal_arity = signal_arity
        self.hox_variant_count = hox_variant_count
        self.counter = counter

        self.input_arity = input_arity
        self.output_arity = output_arity
        self.grid_count = grid_count  # Grids in grid grid per dimension
        self.grid_size = grid_size   # neurons per grid per dimension
        self.actions_max = actions_max
        self.actions_count = 0
        self.input_neurons = []
        middle_grid_input = math.floor(grid_count/2)
        middle_grid_output = math.floor(grid_count/2+1)
        self.init_grids()
        if input_arity > grid_size:
            raise Exception("grid size too small to contain inputs")
        if output_arity > grid_size: 
            raise Exception("grid size too small to contain outputs")
        for num in range(input_arity):
            # think input neurons (sensors) and output neurons
            # (acuators) are close together in the human brain, right?
            # Because they evovled first? 
            # So try to put them close to each other (ex. adjacent grids in the middle)
            self.input_neurons.append(InputNeuron(
                self.grids[middle_grid_input][middle_grid_input][middle_grid_input], 
                middle_grid_input*grid_size + num, middle_grid_input*grid_size + num, 
                middle_grid_input*grid_size + num))
        self.output_neurons = []
        for _ in range(output_arity):
            self.output_neurons.append(OutputNeuron(
                self.grids[middle_grid_output][middle_grid_output][middle_grid_output],
                middle_grid_output*grid_size + num,
                middle_grid_output*grid_size + num,
                middle_grid_output*grid_size + num))
        self.neurons = []
        self.init_neurons(middle_grid_input)
        self.action_queue = []
        self.timestep_indexes = []
        self.debugging = debugging
        self.timestep = 0 
        self.free_connection_grids = [
            self.grids[middle_grid_input][middle_grid_input][middle_grid_input],
            self.grids[middle_grid_output][middle_grid_output][middle_grid_output]
        ]
    
    def get_size_in_neuron_positions_one_dim(self):
        return self.grid_count * self.grid_size

    def get_current_timestep(self):
        return self.timestep

    def get_free_dendrite(self, neuron, target_distance):
        _target_dist = target_distance
        if len(self.free_connection_grids != 0):
            if _target_dist == 0 and neuron.grid in self.free_connection_grids:
                _x = [x for x in neuron.grid.free_dendrites if x not in neuron.dendrites and x not in neuron.axons]
                if len(_x) != 0:
                    return random.choice(_x)
            else: 
                dists = [(x, abs(self.approximate_distance_grid(neuron.grid, x) - _target_dist)) for x in self.free_connection_grids]
                best_match_grid = min(dists, key=lambda x: x[1])[0]
                return random.choice(best_match_grid.free_dendrites)
        return None
    
    # RFE assumes knowledge of neuron class properties
    def update_neuron_position(self, neuron, neuron_pos_loc):
        neuron_grid = neuron.grid
        (xg, yg, zg) = (neuron_grid.x, neuron_grid.y, neuron_grid.z)
        (xl, yl, zl) = neuron_pos_loc
        def _modify_pattern(local_pos, grid_size, global_pos, grid_pos, grid_count):
            if local_pos > grid_size:
                if grid_pos < grid_count:
                    new_local = 0
                    new_grid = grid_pos + 1
                    new_global = global_pos
                else: 
                    new_local = local_pos - 1
                    new_grid = grid_pos
                    new_global = global_pos - 1
            elif local_pos < 0:
                if grid_pos > 0:
                    new_local = grid_size
                    new_grid = grid_pos - 1
                    new_global = global_pos
                else: 
                    new_local = local_pos + 1
                    new_global = global_pos + 1
                    new_grid = grid_pos
            else: 
                new_grid = grid_pos
                new_local = local_pos
                new_global = global_pos
            return new_grid, new_local, new_global

        gridx, localx, globalx = _modify_pattern(xl, self.grid_size, neuron.x_glob, xg, self.grid_count)
        gridy, localy, globaly = _modify_pattern(yl, self.grid_size, neuron.y_glob, yg, self.grid_count)
        gridz, localz, globalz = _modify_pattern(zl, self.grid_size, neuron.z_glob, zg, self.grid_count)

        return self.grids[gridx][gridy][gridz], (localx, localy, localz), (globalx, globaly, globalz)


    def init_grids(self):
        self.grids = {}
        for n1 in range(self.grid_count):
            for n2 in range(self.grid_count):
                for n3 in range(self.grid_count):
                    self.grids["".join(str(n1), str(n2), str(n3))] = Grid(n1, n2, n3, self.grid_size, self)
    
    def init_neurons(self, middle_grid_input):
        self.add_neuron((middle_grid_input, middle_grid_input, middle_grid_input), 
                         [0 for _ in range(self.neuron_initialization_data['internal_state_variable_count'])])
        
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
        # to be used to determine candidate neurons of a given distance, as gridwise distance helps limit search space from all neurons to only ones in 
        # approximate distance grids. 
        return abs(grid1.x - grid2.x) + abs(grid1.y - grid2.y) + abs(grid1.z - grid2.z)
    
    def correct_neuron_distance(self, neuron1, neuron2):
        x0, y0, z0 = neuron1.grid.to_global((neuron1.x, neuron1.y, neuron1.z))
        x1, y1, z1 = neuron2.grid.to_global((neuron2.x, neuron2.y, neuron2.z))
        return (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2
    
    def run_sample(self, input_sample):
        for num in range(self.input_neurons):
            self.input_neurons[num].value = input_sample.value
        self.action_queue = []
        for input_neuron in self.input_neurons:
            for axon in input_neuron.subscribers:
                self.add_action_to_queue(
                    axon.run_recieve_signal_dendrite(
                        [input_neuron.value for _ in range(self.signal_arity)],
                        0
                    ),
                    0,
                    axon.id
                )
        self.timestep = 0
        self.actions_count = 0
        while len(self.action_queue > 0) and self.actions_count < self.actions_max:
            timestep, action, _ = self.action_queue.pop(0)
            action(self)  # runs action
            self.timestep = timestep
            self.actions_count += 1
        outputs = [output_neuron.value for output_neuron in self.output_neurons]
        return outputs

    def run(self, problem):
        # for every sample in inputs run it, some way of evaluating result as well
        # run run_sample for each sample some amount of times
        # then use eval function of problem class to give reward
        # Assumes goal is to minimize error
        cumulative_error = 0
        for num in range(self.instances_per_iteration):
            instance = problem.get_problem_instance()
            result = self.run_sample(instance)
            error = problem.error(instance, result)
            cumulative_error += error
            reward = problem.get_reward(error)
            self.action_queue = []
            self.timestep = 0
            self.actions_count = 0
            for neuron in self.neurons:
                self.add_action_to_queue(
                    neuron.run_recieve_reward(reward, 0),
                    0,
                    neuron.id
                )
                for axon in neuron.dendrites + neuron.axons:
                    self.add_action_to_queue(
                        axon.run_recieve_reward(reward, 0),
                        0,
                        axon.id
                    )
            while len(self.action_queue) > 0:
                timestep, action, _ = self.action_queue.pop(0)
                action()  # runs action
                self.timestep = timestep
                self.actions_count += 1
        return cumulative_error/self.instances_per_iteration

    def add_action_to_queue(self, action, timestep, id):
        timesteps = [x[0] for x in self.timestep_indexes]
        timestep_relative_order = 0
        for x in timesteps:
            if timestep > x: 
                timestep_relative_order += 1
        if timestep_relative_order > len(self.timestep_indexes):
            self.action_queue.append((timestep, action, id))
            self.timestep_indexes.append((timestep, len(self.action_queue-1), id))
        else:
            pos = self.timestep_indexes[timestep_relative_order][1]+1
            self.action_queue.insert(pos, (timestep, action, id))
            self.timestep_indexes.insert((timestep_relative_order, pos, id))
            for x in self.timestep_indexes[timestep_relative_order+1:]:
                x[1] = x[1]+1

    def remove_actions_id(self, id):
        to_remove_indexes = []
        for num in range(len(self.timestep_indexes)):
            _, _, _id = self.timestep_indexes[num]
            if id == _id:
                to_remove_indexes.append(num)
        for val in to_remove_indexes:
            _, action_index, _ = self.timestep_indexes[val]
            self.action_queue.remove(self.action_queue[action_index])
            self.timestep_indexes.remove(self.timestep_indexes[val])

    def remove_neuron(self, neuron):
        nid = neuron.id
        neuron.grid.remove_neuron(neuron)
        self.remove_actions_id(nid)

    def add_neuron(self, neuron_pos, neuron_internal_state):
        grid = self.grids[neuron_pos[0]//self.grid_size][neuron_pos[1]//self.grid_size][neuron_pos[2]//self.grid_size]
        new_neuron = Neuron(
            neuron_initialization_data = self.neuron_initialization_data, 
            axon_initialization_data = self.axon_initialization_data,
            neuron_engine = self,
            x_glob = neuron_pos[0],
            y_glob = neuron_pos[1],
            z_glob = neuron_pos[2],
            x_loc = neuron_pos[0] % self.grid_size,
            y_loc = neuron_pos[1] % self.grid_size,
            z_loc = neuron_pos[2] % self.grid_size,
            signal_arity = self.signal_arity,
            hox_variants = self.hox_variant_count,
            counter = self.counter,
            grid = grid
        )
        grid.add_neuron(new_neuron)
        self.neurons.append(new_neuron)
    
    def remove_dendrite(self, dendrite):
        self.remove_actions_id(dendrite.id)

class Grid():
    def __init__(self, x, y, z, size, neuron_engine) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.neurons = []
        self.free_dendrites = []
        self.neuron_engine = neuron_engine
    
    def contains(self, neuron):
        return neuron.id in [x.id for x in self.neurons]
    
    def to_global(self, pos):
        # pos = (x, y, z)
        g_x = pos[0] + self.x*self.size
        g_y = pos[1] + self.y*self.size
        g_z = pos[2] + self.z*self.size
        return (g_x, g_y, g_z)

    def add_free_dendrite(self, dendrite):
        self.free_dendrites.append(dendrite)
        if self.free_dendrites.count == 1:
            self.neuron_engine.free_connection_grids.append(self)

    def remove_free_dendrite(self, dendrite):
        if dendrite in self.free_dendrites:
            self.free_dendrites.remove(dendrite)
            if self.free_dendrites.count == 0:
                self.neuron_engine.free_connection_grids.remove(self)
    
    def remove_neuron(self, neuron):
        if neuron in self.neurons:
            self.neurons.remove(neuron)
    
    def add_neuron(self, neuron):
        if neuron not in self.neurons:
            self.neurons.append(neuron)
               

class InputNeuron(Neuron):
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in
        super(Neuron, self).__init__(grid)
        self.x = x
        self.y = y
        self.z = z
        self.subscribers = []
        self.value = None
    
    def run_accept_connection(self, dendrite, timestep):
        return not dendrite in self.subscribers
    
    def add_subscriber(self, target):
        if target not in self.subscribers:
            self.subscribers.append(target)
    
    def remove_subscriber(self, target):
        self.subscribers.remove(target)

class OutputNeuron(Neuron):
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in 
        super(Neuron, self).__init__(grid)
        self.x = x
        self.y = y
        self.z = z
        self.subscribers = []
        self.value = None

    def run_accept_connection(self, dendrite, timestep):
        return dendrite not in self.subscribers
    
    def add_subscriber(self, target):
        if target not in self.subscribers:
            self.subscribers.append(target)
    
    def remove_subscriber(self, target):
        self.subscribers.remove(target)

    def run_recieve_signal_axon(self, signals, timestep):
        self.value = signals[0] 