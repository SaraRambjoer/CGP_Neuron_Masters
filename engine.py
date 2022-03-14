# here define the world space
from HelperClasses import randchoice, randcheck, listmult, Counter
import math
import numpy as np
import warnings

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
                 instances_per_iteration,
                 logger,
                 genome_id,
                 config_file,
                 debugging = False):


        self.config = config_file

        self.counter = Counter()
        self.genome_id = genome_id

        self.logger = logger
        self.instances_per_iteration = instances_per_iteration
        self.neuron_initialization_data = neuron_initialization_data
        self.axon_initialization_data = axon_initialization_data
        self.signal_arity = signal_arity
        self.hox_variant_count = hox_variant_count
        self.changed = False
        self.not_changed_count = 0

        self.hox_switches_diagnostic = 0

        
        self.input_arity = input_arity
        self.output_arity = output_arity
        self.grid_count = grid_count  # Grids in grid grid per dimension
        self.grid_size = grid_size   # neurons per grid per dimension
        self.actions_max = actions_max
        self.actions_count = 0  # For counting how many actions engine is allowed to take
        self.input_neurons = []
        self.middle_grid_input = math.floor(grid_count/2)
        self.middle_grid_output = math.floor(grid_count/2+1)
        self.init_grids()
        if input_arity > grid_size:
            raise Exception("grid size too small to contain inputs")
        if output_arity > grid_size: 
            raise Exception("grid size too small to contain outputs")
        self.neurons = []
        self.logger.log("run_start", (f"{self.genome_id.split('->')[1][1:]}-setup", "setup"))
        self.logger.log("instance_start", "setup")
        self.init_neurons()
        self.reset()
        self.action_queue = []
        self.debugging = debugging
        self.timestep = 0 
    
    def get_neuron_count(self):
        return len(self.neurons)
    
    def get_average_hidden_connectivity(self):
        connection_count = 0
        for neuron in self.neurons:
            connection_count += len([x for x in neuron.axons + neuron.dendrites if x.connected_dendrite is not None])
        return connection_count/len(self.neurons)
    
    def get_output_connectivity(self, unique=False):
        neurons = []
        for neuron in self.output_neurons:
            neurons += [x.neuron for x in neuron.subscribers]
        if unique: 
            return len(set([x.id for x in neurons]))
        return len(neurons)/len(self.output_neurons)
    
    def get_input_connectivity(self, unique=False):
        neurons = []
        for neuron in self.input_neurons:
            neurons += [x.neuron for x in neuron.subscribers]
        if unique: 
            return len(set([x.id for x in neurons]))
        return len(neurons)/len(self.input_neurons)            

    
    def get_size_in_neuron_positions_one_dim(self):
        return self.grid_count * self.grid_size

    def get_current_timestep(self):
        return self.timestep

    def get_free_dendrite(self, neuron, target_distance, from_axon):
        _target_dist = target_distance // self.grid_size
        if len(self.free_connection_grids) != 0:
            if _target_dist == 0 and neuron.grid in self.free_connection_grids:
                chosen_grid = neuron.grid
            else: 
                dists = [(x, abs(self.approximate_distance_grid(neuron.grid, x) - _target_dist)) for x in self.free_connection_grids]
                chosen_grid = min(dists, key=lambda x: x[1])[0]
            _x = [x for x in chosen_grid.free_dendrites if x not in neuron.dendrites and x not in neuron.axons]
            if from_axon:
                _x = [x for x in _x if x not in x.neuron.axons]
            else:
                _x = [x for x in _x if x not in x.neuron.dendrites]
            input_and_output_neurons = [x for x in chosen_grid.neurons if type(x) == InputNeuron or type(x) == OutputNeuron]
            _x += input_and_output_neurons
            if len(_x) != 0:
                return randchoice(_x)

        return None
    


    # RFE assumes knowledge of neuron class properties
    def update_neuron_position(self, neuron, neuron_pos_loc):
        self.changed = True
        neuron_grid = neuron.grid
        (xg, yg, zg) = (neuron_grid.x, neuron_grid.y, neuron_grid.z)
        (xl, yl, zl) = neuron_pos_loc
        def _modify_pattern(local_pos, grid_size, global_pos, grid_pos, grid_count):
            if local_pos >= grid_size:
                if grid_pos < grid_count-1:
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

        while globalx >= self.grid_count*self.grid_size:
            globalx -= 1
            localx -= 1
            if localx < 0:
                gridx -= 1
                localx = self.grid_size-1
        while globaly >= self.grid_count*self.grid_size:
            globaly -= 1
            localy -= 1
            if localy < 0:
                gridy -= 1
                localy = self.grid_size-1
        while globalz >= self.grid_count*self.grid_size:
            globalz -= 1
            localz -= 1
            if localz < 0:
                gridz -= 1
                localz = self.grid_size-1
        return self.grids[gridx][gridy][gridz], (localx, localy, localz), (globalx, globaly, globalz)


    def init_grids(self):
        self.grids = [[[None for _ in range(self.grid_count)] for _ in range(self.grid_count)] for _ in range(self.grid_count)]
        for n1 in range(self.grid_count):
            for n2 in range(self.grid_count):
                for n3 in range(self.grid_count):
                    self.grids[n1][n2][n3] = Grid(n1, n2, n3, self.grid_size, self, self.logger)
        self.free_connection_grids = [
            self.grids[self.middle_grid_input][self.middle_grid_input][self.middle_grid_input],
            self.grids[self.middle_grid_output][self.middle_grid_output][self.middle_grid_output]
        ]

    def get_neuron_auto_connections(self):
        neuron_auto_connections = 0
        for neuron in self.neurons:
            for connection in neuron.axons + neuron.dendrites:
                if type(connection) is not InputNeuron and type(connection) is not OutputNeuron:
                    if connection.connected_dendrite is not None and type(connection.connected_dendrite) is Axon:
                        if connection.neuron == connection.connected_dendrite.neuron:
                            neuron_auto_connections += 1
        return neuron_auto_connections



    def init_neurons(self):
        self.add_neuron((self.middle_grid_input, self.middle_grid_input, self.middle_grid_input), 
                         [0 for _ in range(self.neuron_initialization_data['internal_state_variable_count'])])
        for num in range(self.input_arity):
            # think input neurons (sensors) and output neurons
            # (acuators) are close together in the human brain, right?
            # Because they evovled first? 
            # So try to put them close to each other (ex. adjacent grids in the middle)
            self.input_neurons.append(InputNeuron(
                self.grids[self.middle_grid_input][self.middle_grid_input][self.middle_grid_input], 
                self.middle_grid_input*self.grid_size + num, self.middle_grid_input*self.grid_size + num, 
                self.middle_grid_input*self.grid_size + num, 
                self.logger, 
                self.counter,
                self))
        self.output_neurons = []
        for _ in range(self.output_arity):
            self.output_neurons.append(OutputNeuron(
                self.grids[self.middle_grid_output][self.middle_grid_output][self.middle_grid_output],
                self.middle_grid_output*self.grid_size + num,
                self.middle_grid_output*self.grid_size + num,
                self.middle_grid_output*self.grid_size + num,
                self.logger, 
                self.counter,
                self))

        
    def reset(self):
        self.actions_count = 0
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []
        self.grids = []
        self.init_grids()
        self.init_neurons()
        self.timestep = 0
        self.hox_switches_diagnostic = 0

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
    
    def run_sample(self, input_sample, action_counts):
        self.action_counts = action_counts
        for num in range(len(self.input_neurons)):
            self.input_neurons[num].value = input_sample[num]
        self.action_queue = []
        axon_ids = []
        for input_neuron in self.input_neurons:
            inner_axon_ids = []
            for axon in input_neuron.subscribers:
                if axon.id in axon_ids:
                    raise Exception(f"Axon id {axon.id} connected to several input neurons. Genome id: {self.genome_id}")
                elif axon.id in inner_axon_ids:
                    raise Exception(f"Axon id {axon.id} connected to same Input Neuron several times. Genome id: {self.genome_id}")
                self.add_action_to_queue(
                    lambda: axon.run_recieve_signal_axon(
                        [input_neuron.value for _ in range(self.signal_arity)],
                        0
                    ),
                    0,
                    axon.id,
                    'dendrite_recieve_signal_axon'
                )
                inner_axon_ids.append(axon.id)
            axon_ids += inner_axon_ids
        self.timestep = 0
        self.actions_count = 0
        neuron_ids = []
        for neuron in self.neurons:
            if neuron.id in neuron_ids:
                raise Exception(f"Neuron id {neuron.id} in neurons list several times")
            self.add_action_to_queue(
                lambda: neuron.run_action_controller(0),
                0,
                neuron.id,
                'neuron_action_controller'
            )
            neuron_ids.append(neuron)
        while len(self.action_queue) > 0 and self.actions_count < self.actions_max:
            action, timestep, _, name = self.action_queue.pop()
            if not action == "skip":
                action()  # runs action
                self.timestep = timestep
                self.actions_count += 1
                self.action_counts[name] += 1
        outputs = [output_neuron.value for output_neuron in self.output_neurons]
        self.reset_outputs()
        return outputs

    def reset_outputs(self):
        for output_neuron in self.output_neurons: 
            output_neuron.value = None

    def run(self, problem, runnum):
        # for every sample in inputs run it, some way of evaluating result as well
        # run run_sample for each sample some amount of times
        # then use eval function of problem class to give reward
        # Assumes goal is to minimize error
        self.logger.log("run_start", (self.genome_id.split("->")[1][1:], runnum))
        self.reset()
        cumulative_error = 0
        exec_instances = 0

        # Dendrite/Axon distinction is only made to talk about program flow
        self.action_counts = {
            'neuron_axon_birth':0,
            'neuron_dendrite_birth':0,
            'neuron_signal_axon':0,
            'neuron_signal_dendrite':0,
            'neuron_recieve_axon_signal':0,
            'neuron_recieve_reward':0,
            'neuron_move':0,
            'neuron_die':0,
            'neuron_neuron_birth':0,
            'neuron_action_controller':0,
            'neuron_hox_variant_selection':0,
            'dendrite_recieve_signal_neuron':0,
            'axon_recieve_signal_neuron':0,
            'dendrite_recieve_signal_dendrite':0,
            'dendrite_recieve_signal_axon':0,
            'axon_recieve_signal_dendrite':0,
            'dendrite_signal_axon':0,
            'axon_signal_dendrite':0,
            'dendrite_signal_dendrite':0,
            'dendrite_signal_neuron':0,
            'dendrite_accept_connection':0,
            'dendrite_break_connection':0,
            'dendrite_recieve_reward':0,
            'dendrite_die':0,
            'dendrite_action_controller':0,
            'dendrite_seek_connection':0, 
            'dendrite_axon_death_connection_signal':0,
            'dendrite_axon_death_neuron_signal':0,
            'skip_post_death':0
        }

        base_problems = {"no outputs" : 0,
                        "no neurons" : 0,
                        "no input connections" : 0,
                        "no output connections" : 0,
                        "no connections" : 0}

        smooth_grad = self.config['smooth_gradient']
        for num in range(self.instances_per_iteration):
            self.logger.log("instance_start", num)
            graphlog_initial_data = {"run number": runnum, "iteration": num, "genome_id" : self.genome_id}
            self.graph_log("graphlog_instance", graphlog_initial_data)
            exec_instances += 1
            self.changed = False
            instance = problem.get_problem_instance()
            result = self.run_sample(instance, self.action_counts)
            error, valid_output = problem.error(instance, result, self.logger)
            if not valid_output:
                base_problems["no outputs"] += 1
            else: 
                self.not_changed_count = 0  # No need to change if it is working
            if len(self.neurons) == 0:
                base_problems["no neurons"] += 1
            no_input_connections = True
            # Always bad
            for node in self.input_neurons:
                if len(node.subscribers) != 0:
                    no_input_connections = False
                    break
            if no_input_connections:
                base_problems["no input connections"] += 1
            no_output_connections = True
            for node in self.output_neurons:
                if len(node.subscribers) != 0:
                    no_output_connections = False
                    break
            if no_output_connections:
                base_problems["no output connections"] += 1
        
            no_neuron_connections = True
            for node in self.neurons:
                if len(node.axons) != 0 or len(node.dendrites) != 0:
                    no_neuron_connections = False
            if no_neuron_connections:
                base_problems["no connections"] += 1
            
            cumulative_error += error
            reward = problem.get_reward(error)
            self.logger.log("instance_results", f"{error}, {reward}")
            self.action_queue = []
            self.timestep = 0
            self.actions_count = 0
            self.logger.log("reward_phase", "reward phase begins")
            for neuron in self.neurons:
                self.add_action_to_queue(
                    lambda: neuron.run_recieve_reward(reward, 0),
                    0,
                    neuron.id,
                    'neuron_recieve_reward'
                )
                for axon in neuron.dendrites + neuron.axons:
                    self.add_action_to_queue(
                        lambda: axon.run_recieve_reward(reward, 0),
                        0,
                        axon.id,
                        'dendrite_recieve_reward'
                    )
            while len(self.action_queue) > 0 and self.actions_count < self.actions_max:
                action, timestep, _, name = self.action_queue.pop()
                if not action == "skip":
                    action()  # runs action
                    self.timestep = timestep
                    self.actions_count += 1
                    self.action_counts[name] += 1
            if not self.changed:
                if self.not_changed_count > 10:
                    base_problems = {"Samples tried": exec_instances, **base_problems}
                    if smooth_grad:
                        cumulative_error += (self.instances_per_iteration - num - 1)*(1+1/self.instances_per_iteration)
                    else:
                        cumulative_error += (self.instances_per_iteration - num - 1)
                    exec_instances = self.instances_per_iteration
                    break
                else:
                    self.not_changed_count += 1
            else:
                self.not_changed_count = 0
            self.logger.log("instance_end", "no_message")
        
        
        base_problems["neuron_count"] = self.get_neuron_count()
        base_problems["neuron_connectivity"] = self.get_average_hidden_connectivity()
        base_problems["input_connectivity"] = self.get_input_connectivity()
        base_problems["nodes_connected_to_input_nodes"] = self.get_input_connectivity(True)
        base_problems["output_connectivity"] = self.get_output_connectivity()
        base_problems["nodes_connected_to_output_nodes"] = self.get_output_connectivity(True)
        base_problems["hox_switch_count"] = int(str(self.hox_switches_diagnostic))
        base_problems["neuron_auto_connectivity"] = self.get_neuron_auto_connections()

        if smooth_grad:
            # node count penalty: 
            cumulative_error += max((3-base_problems['neuron_count'])/3, 0)
            # node connectivity penalty
            cumulative_error += max((3-base_problems['neuron_connectivity'])/3, 0)
            # input connectivity penalty:
            cumulative_error += max(1-base_problems['input_connectivity'], 0)
            # output connectivity penalty
            cumulative_error += max(1-base_problems['output_connectivity'], 0)

        self.graph_log("graphlog_run", graphlog_initial_data)
        self.logger.log("run_end", f"{cumulative_error}, {base_problems}")
        return cumulative_error/exec_instances, base_problems, self.action_counts


    def add_action_to_queue(self, action, timestep, id, action_name):
        if timestep == self.timestep:
            self.action_queue.insert(0, (action, timestep, id, action_name))
        else:
            self.action_queue.append((action, timestep, id, action_name))
    

    def remove_actions_id(self, id):
        for num in range(len(self.action_queue)):
            current = self.action_queue[num]
            if current[2] == id:
                self.action_queue[num] = ("skip", current[1], current[2], "skip_post_death")
            

    def remove_neuron(self, neuron):
        self.changed = True
        nid = neuron.id
        neuron.grid.remove_neuron(neuron)
        self.remove_actions_id(nid)

    def _grid_cap(self, pos):
        pos = int(math.floor(pos))
        if pos > self.grid_count-1:
            warnings.warn("Position", pos, "is greater than grid count - 1", self.grid_count-1, "which should not be possible. Handling error by capping value.")
        return max(pos, self.grid_count-1)

    def add_neuron(self, neuron_pos, neuron_internal_state):
        self.changed = True
        grid = self.grids[self._grid_cap(neuron_pos[0]//self.grid_size)][self._grid_cap(neuron_pos[1]//self.grid_size)][self._grid_cap(neuron_pos[2]//self.grid_size)]
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
            grid = grid,
            logger = self.logger,
            config = self.config
        )
        new_neuron.internal_states = neuron_internal_state
        grid.add_neuron(new_neuron)
        self.neurons.append(new_neuron)

    def add_neuron_created(self, neuron):
        self.changed = True
        neuron_pos = (neuron.x_glob, neuron.y_glob, neuron.z_glob)
        grid = self.grids[self._grid_cap(neuron_pos[0]//self.grid_size)][self._grid_cap(neuron_pos[1]//self.grid_size)][self._grid_cap(neuron_pos[2]//self.grid_size)]
        neuron.grid = grid
        grid.add_neuron(neuron)
        self.neurons.append(neuron)
    
    def remove_dendrite(self, dendrite):
        self.changed = True
        self.remove_actions_id(dendrite.id)
    
    def graph_log(self, message_type, initial_data):
        neuron_id_list = [x.id for x in self.neurons + self.input_neurons + self.output_neurons]
        input_id_list = [x.id for x in self.input_neurons]
        output_id_list = [x.id for x in self.output_neurons]
        neuron_id_to_pos = [(neuron.id, (neuron.x_glob, neuron.y_glob, neuron.z_glob)) for neuron in self.neurons + self.input_neurons + self.output_neurons]
        connections_dendritewise = []
        for neuron in self.neurons: 
            from_id = neuron.id
            for dendrite in neuron.dendrites:
                if not dendrite.connected_dendrite is None:
                    if type(dendrite.connected_dendrite) in [InputNeuron, OutputNeuron]:
                        candidate = (from_id, dendrite.connected_dendrite.id)
                    else:
                        candidate = (from_id, dendrite.connected_dendrite.neuron.id)
                    connections_dendritewise.append(candidate)
            for axon in neuron.axons:
                if not axon.connected_dendrite is None:
                    if type(axon.connected_dendrite) in [InputNeuron, OutputNeuron]:
                        candidate = (axon.connected_dendrite.id, from_id)
                    else:
                        candidate = (axon.connected_dendrite.neuron.id, from_id)
                    if candidate not in connections_dendritewise:
                        connections_dendritewise.append(candidate)
        log_json = {
            "neurons": neuron_id_list, 
            "inputs": input_id_list, 
            "outputs": output_id_list, 
            "connections": connections_dendritewise,
            "neuron_id_to_pos" : neuron_id_to_pos,
            **initial_data}
        self.logger.log_json(message_type, log_json)


class Grid():
    def __init__(self, x, y, z, size, neuron_engine, logger) -> None:
        self.logger = logger
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
        self.neuron_engine.changed = True
        self.free_dendrites.append(dendrite)
        self.neuron_engine.changed = True
        if self.free_dendrites.count == 1:
            self.neuron_engine.free_connection_grids.append(self)

    def remove_free_dendrite(self, dendrite):
        self.neuron_engine.changed = True
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

               

class InputNeuron():
    def __init__(self, grid, x, y, z, logger, counter, engine) -> None:
        # should know which grid it is in
        self.logger = logger
        self.grid = grid
        grid.add_neuron(self)
        self.x = x
        self.y = y
        self.z = z
        self.x_glob = x
        self.y_glob = y
        self.z_glob = z
        self.subscribers = []
        self.value = None
        self.id = counter.counterval()
        self.engine = engine

    
    def run_accept_connection(self, dendrite, timestep):
        if not dendrite in self.subscribers:
            self.logger.log("engine_action", f"{self.id}, {timestep}: input neuron accepts connection, dendrite {dendrite.id} not in connection list")
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: input neuron rejects connection, dendrite {dendrite.id} in connection list")
        return not dendrite in self.subscribers
    
    def add_subscriber(self, target):
        self.subscribers.append(target)
    
    def remove_subscriber(self, target):
        if target in self.subscribers:
            self.subscribers.remove(target)

class OutputNeuron():
    def __init__(self, grid, x, y, z, logger, counter, engine) -> None:
        # should know which grid it is in 
        self.logger = logger
        self.grid = grid
        grid.add_neuron(self)
        self.x = x
        self.y = y
        self.z = z
        self.x_glob = x
        self.y_glob = y
        self.z_glob = z
        self.subscribers = []
        self.value = None
        self.engine = engine
        self.id = counter.counterval()

    def run_accept_connection(self, dendrite, timestep):
        if not dendrite in self.subscribers:
            self.logger.log("engine_action", f"{self.id}, {timestep}: output neuron accepts connection, dendrite {dendrite.id} not in connection list")
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: output neuron rejects connection, dendrite {dendrite.id} in connection list")
        return dendrite not in self.subscribers
    
    def add_subscriber(self, target):
        self.subscribers.append(target)
    
    def remove_subscriber(self, target):
        if target in self.subscribers:
            self.subscribers.remove(target)

    def run_recieve_signal_axon(self, signals, timestep):
        self.value = signals[0] 



def addqueue(neuron_engine, lambdafunc, timestep, id, action_name):
    neuron_engine.add_action_to_queue(
        lambdafunc,
        timestep,
        id,
        action_name
    )


class CellObjectSuper:
    def __init__(self, logger, config):
        self.x_glob = None
        self.y_glob = None
        self.z_glob = None
        self.logger = logger
        self.config = config

    def set_global_pos(self, program, indexes):
        if type(self) is Neuron:
            program.input_nodes[indexes[0]].set_output(self.x_glob)
            program.input_nodes[indexes[1]].set_output(self.y_glob)
            program.input_nodes[indexes[2]].set_output(self.z_glob)
        elif type(self) is Axon:
            program.input_nodes[indexes[0]].set_output(self.neuron.x_glob)
            program.input_nodes[indexes[1]].set_output(self.neuron.y_glob)
            program.input_nodes[indexes[2]].set_output(self.neuron.z_glob)
    
    def set_internal_state_inputs(self, program):
        num = 0
        initial = program.get_input_none_index()
        for input_node in program.input_nodes[initial:initial+self.internal_state_variable_count]:
            input_node.set_output(self.internal_states[num])
            num += 1
        num = 0
        for input_node in program.input_nodes[initial+self.internal_state_variable_count:\
            initial+self.internal_state_variable_count+len(self.config['cgp_function_constant_numbers'])]:
            input_node.set_output(self.config['cgp_function_constant_numbers'][num])
            num += 1
    
    def update_internal_state(self, deltas):
        old_internal_state = [x for x in self.internal_states]
        for num in range(self.internal_state_variable_count):
            newval = deltas[num] + self.internal_states[num]
            self.internal_states[num] = np.amax([np.amin([newval, 100.0]), -100.0]) 
            # Crop range of number values internal states can take, otherwise may run into overflow in cgp engine
        self.logger.log("engine_action", f"{id}: Updated internal state: {old_internal_state} -> {self.internal_states}")






# TODO change action controller s.t. it can generate and send signals?
# RFE some code duplication issues
# also there is going ot be some knowledge duplication in terms of length of inputs and outputs for each program
class Neuron(CellObjectSuper):
    def __init__(self, 
                neuron_initialization_data,
                axon_initialization_data,
                neuron_engine,
                x_glob,
                y_glob,
                z_glob,
                x_loc,
                y_loc,
                z_loc,
                signal_arity,
                hox_variants,
                counter,
                grid,
                logger,
                config) -> None:
        super(Neuron, self).__init__(logger, config)
        self.neuron_initialization_data = neuron_initialization_data
        self.axon_initialization_data = axon_initialization_data
        self.config = config
        self.id = counter.counterval()

        self.hox_variant_selection_program = self.neuron_initialization_data['hox_variant_selection_program']
        self.internal_state_variable_count = self.neuron_initialization_data['internal_state_variable_count']

        self.axon_birth_program = None
        self.signal_axon_program = None
        self.recieve_axon_signal_program = None
        self.recieve_reward_program = None
        self.move_program = None
        self.die_program = None
        self.neuron_birth_program = None
        self.action_controller_program = None
        self.axons = []
        self.dendrites = []
        self.cgp_constant_count = len(self.config['cgp_function_constant_numbers'])

        self.counter = counter

        self.internal_states = []
        for _ in range(self.internal_state_variable_count):
            self.internal_states.append(0.0)
        
        self.neuron_engine = neuron_engine

        self.dying = False

        self.x_glob = x_glob
        self.y_glob = y_glob
        self.z_glob = z_glob
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.z_loc = z_loc

        self.signal_arity = signal_arity

        self.hox_variants = hox_variants
        self.selected_hox_variant = 0
        self.run_hox_selection()
        self.grid = grid
        self.program_order_length = 7
    

    def program_order_lambda_factory(self, timestep, index):
        if index == 0:
            return lambda: self.run_dendrite_birth(timestep), "neuron_dendrite_birth"
        elif index == 1:
            return lambda: self.run_axon_birth(timestep), 'neuron_axon_birth'
        elif index == 2:
            return lambda: self.run_move(timestep), 'neuron_move'
        elif index == 3:
            return lambda: self.run_die(timestep), 'neuron_die'
        elif index == 4:
            return lambda: self.run_neuron_birth(timestep), 'neuron_neuron_birth'
        elif index == 5:
            return lambda: self.run_hox_selection(), "neuron_hox_variant_selection"
        elif index == 6:
            return lambda: self.run_action_controller(timestep), "neuron_action_controller"
        else:
            raise Exception("Invalid program order index")

    def load_hox_programs(self):
        self.axon_birth_program = self.neuron_initialization_data['axon_birth_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.signal_axon_program = self.neuron_initialization_data['signal_axon_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.recieve_axon_signal_program = self.neuron_initialization_data['recieve_axon_signal_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.recieve_reward_program = self.neuron_initialization_data['recieve_reward_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.move_program = self.neuron_initialization_data['move_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.die_program = self.neuron_initialization_data['die_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.neuron_birth_program = self.neuron_initialization_data['neuron_birth_programs']\
            .hex_variants[self.selected_hox_variant].program
        self.action_controller_program = self.neuron_initialization_data['action_controller_programs']\
            .hex_variants[self.selected_hox_variant].program

        for axon_dendrite in self.axons + self.dendrites:
            axon_dendrite.load_hox_programs()

    def addqueue_inner(self, lambdafunc, timestep, action_name):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id, action_name)

    def create_new_dendrite(self, dendrite=True):
        new_dendrite = Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self, self.logger, self.config, False)
        if dendrite:
            self.dendrites.append(new_dendrite)
        else:
            self.axons.append(new_dendrite)
        return new_dendrite


    def remove_dendrite(self, dendrite):
        if dendrite in self.axons:
            self.axons.remove(dendrite)
        if dendrite in self.dendrites:
            self.dendrites.remove(dendrite)
        self.neuron_engine.remove_dendrite(dendrite)

    def set_signal_inputs(self, program, signals, offset=0):
        if len(signals) != self.signal_arity:
            raise Exception("Invalid signal dimensionality")
        for num in range(len(signals)):
            program.input_nodes[num+offset].set_output(signals[num])

    def run_axon_birth(self, timestep):
        if not self.dying:
            self.axon_birth_program.reset()
            self.axon_birth_program.input_nodes[0].set_output(len(self.axons))
            self.set_global_pos(self.axon_birth_program, (1, 2, 3))
            self.set_internal_state_inputs(self.axon_birth_program)
            log_inputs = [x.output for x in self.axon_birth_program.input_nodes]
            outputs = self.axon_birth_program.run_presetinputs()

            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran axon birth. Inputs: {log_inputs}. Outputs: {outputs}.")
            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if randcheck(outputs[0]):
                self.axons.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self, self.logger, self.config))
                axon = self.axons[-1]
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran axon birth, axon {axon.id} born. Adding axon action controller to queue.")
                self.addqueue_inner(
                    lambda: axon.run_action_controller(timestep+1), 
                    timestep + 1,
                    "dendrite_action_controller"
                )
            
            if randcheck(outputs[1]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran axon birth, adding signal to axon and dendrites to queue.")
                self.addqueue_inner(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_axon'
                )
                self.addqueue_inner(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_dendrite'
                )
            
            self.update_internal_state(listmult(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count], outputs[2+self.signal_arity]))
            
            if randcheck(outputs[-1]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran axon birth, adding own action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1),
                    timestep + 1,
                    'neuron_action_controller'
                )
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron axon birth aborted, neuron dying.")
    
    def run_dendrite_birth(self, timestep):
        if not self.dying:
            self.axon_birth_program.reset()
            self.axon_birth_program.input_nodes[0].set_output(len(self.dendrites))
            self.set_global_pos(self.axon_birth_program, (1, 2, 3))
            self.set_internal_state_inputs(self.axon_birth_program)
            log_inputs = [x.output for x in self.axon_birth_program.input_nodes]
            outputs = self.axon_birth_program.run_presetinputs()
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran dendrite birth program. Inputs: {log_inputs}. Outputs: {outputs}")

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if randcheck(outputs[0]):
                self.dendrites.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self, self.logger, self.config))
                dendrite = self.dendrites[-1]
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran dendrite birth program, gave birth to dendrite {dendrite.id}. Adding dendrite action controller to queue.")
                self.addqueue_inner(
                    lambda: dendrite.run_action_controller(timestep + 1),
                    timestep + 1,
                    'dendrite_action_controller'
                )
                
            if randcheck(outputs[1]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran dendrite birth program. Adding run_signal to axon and dendrites to queue.")
                self.addqueue_inner(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_axon'
                )
                self.addqueue_inner(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_dendrite'
                )
            
            self.update_internal_state(listmult(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count], outputs[2+self.signal_arity]))
            
            if randcheck(outputs[-1]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran dendrite birth program. Adding own action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep+1),
                    timestep + 1,
                    'neuron_action_controller'
                )
        else: 
            self.logger.log("engine_action", f"{id}, {timestep}: Dendrite birth aborted, neuron dying.")


    def run_signal_dendrite(self, signals, timestep, same_timestep = False):
        self.signal_axon_program.reset()
        self.set_global_pos(self.signal_axon_program, (0, 1, 2))
        self.set_signal_inputs(self.signal_axon_program, signals, 3)
        self.set_internal_state_inputs(self.signal_axon_program)
        log_inputs = [x.output for x in self.signal_axon_program.input_nodes]
        outputs = self.signal_axon_program.run_presetinputs()
        self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal dendrites. Input: {log_inputs}. Outputs: {outputs}")
        if randcheck(outputs[0]):
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal dendrites. Recieve signal in dendrites {[x.id for x in self.dendrites]} added to queue.")
            for dendrite in self.dendrites:
                if same_timestep:
                    self.addqueue_inner(
                        lambda: dendrite.run_recieve_signal_neuron(
                            outputs[1:1+self.signal_arity], 
                            timestep
                        ), 
                        timestep,
                        'dendrite_recieve_signal_neuron'
                    )
                else:
                    self.addqueue_inner(
                        lambda: dendrite.run_recieve_signal_neuron(
                            outputs[1:1+self.signal_arity], 
                            timestep + 1
                        ), 
                        timestep + 1,
                        'dendrite_recieve_signal_neuron'
                    )
        
        self.update_internal_state(listmult(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count], outputs[1+self.signal_arity]))
        
        if randcheck(outputs[-1]): # TODO this one has too few outputs, this output also has another function, check all functions after juleferie
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal dendrites. Adding own action controller to queue.")
            self.addqueue_inner(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1,
                'neuron_action_controller'
            )

    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying: 
            self.recieve_axon_signal_program.reset()
            self.set_global_pos(self.recieve_axon_signal_program, (0, 1, 2))
            self.set_signal_inputs(self.recieve_axon_signal_program, signals, 3)
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()
            log_inputs = [x.output for x in self.recieve_axon_signal_program.input_nodes]

            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal from dendrite. Adding own signal programs to axon and dendrites to queue. Inputs: {log_inputs}. Outputs: {outputs}")

            self.addqueue_inner(
                lambda: self.run_signal_dendrite(outputs[1:self.signal_arity+1], timestep + 1), 
                timestep + 1,
                'neuron_signal_dendrite'
            )

            self.addqueue_inner(
                lambda: self.run_signal_axon(outputs[1:self.signal_arity+1], timestep + 1), 
                timestep + 1,
                'neuron_signal_axon'
            )


            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal from dendrite, added own action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'neuron_action_controller'
                )
            
            self.update_internal_state(listmult(outputs[1+self.signal_arity:1+self.internal_state_variable_count+self.signal_arity], outputs[1+self.signal_arity+self.internal_state_variable_count]))
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal from dendrite, cancelled due to neuron dying. ")

    def run_signal_axon(self, signals, timestep, same_timestep = False):
        self.signal_axon_program.reset()
        self.set_global_pos(self.signal_axon_program, (0, 1, 2))
        self.set_signal_inputs(self.signal_axon_program, signals, 3)
        self.set_internal_state_inputs(self.signal_axon_program)
        log_inputs = [x.output for x in self.signal_axon_program.input_nodes]
        outputs = self.signal_axon_program.run_presetinputs()

        self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal axon. Inputs: {log_inputs}. Outputs: {outputs}")

        if randcheck(outputs[0]):
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal axon. Recieve signal in axons {[x.id for x in self.axons]} added to queue.")
            for axon in self.axons:
                if same_timestep:
                    self.addqueue_inner(
                        lambda: axon.run_recieve_signal_neuron(
                            outputs[1 + self.internal_state_variable_count:1+self.internal_state_variable_count+self.signal_arity], 
                            timestep
                        ), 
                        timestep,
                        'axon_recieve_signal_neuron'
                    )
                else:
                    self.addqueue_inner(
                        lambda: axon.run_recieve_signal_neuron(
                            outputs[1 + self.internal_state_variable_count:1+self.internal_state_variable_count+self.signal_arity], 
                            timestep + 1
                        ), 
                        timestep + 1,
                        'axon_recieve_signal_neuron'
                    )
        
        self.update_internal_state(listmult(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count], outputs[1+self.signal_arity]))
        
        if randcheck(outputs[-1]):
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran signal axon. Added own action controller to queue.")
            self.addqueue_inner(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1,
                'neuron_action_controller'
            )

    def run_recieve_signal_axon(self, signals, timestep):
        if not self.dying:
            self.recieve_axon_signal_program.reset()
            self.set_global_pos(self.recieve_axon_signal_program, (0, 1, 2))
            self.set_signal_inputs(self.recieve_axon_signal_program, signals, 3)
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            log_inputs = [x.id for x in self.recieve_axon_signal_program.input_nodes]
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal axon. Adding own signal programs to dendrites and axons to queue. Inputs: {log_inputs}. Outputs: {outputs}.")

            self.update_internal_state(listmult(outputs[1+self.signal_arity:1+self.signal_arity+self.internal_state_variable_count], outputs[1+self.signal_arity+self.internal_state_variable_count]))

            self.addqueue_inner(
                lambda: self.run_signal_dendrite(outputs[1:1+self.signal_arity], timestep + 1), 
                timestep + 1,
                'neuron_signal_dendrite'
            )

            self.addqueue_inner(
                lambda: self.run_signal_axon(outputs[1:1+self.signal_arity], timestep + 1), 
                timestep + 1,
                'neuron_signal_axon'
            )


            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal axon, added own action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'neuron_action_controller'
                )
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal axon, but cancelled due to neuron dying.")

    def run_recieve_reward(self, reward, timestep):
        if not self.dying:
            self.recieve_reward_program.reset()
            self.recieve_reward_program.input_nodes[0].set_output(reward)
            self.set_global_pos(self.recieve_reward_program, (1, 2, 3))
            self.set_internal_state_inputs(self.recieve_reward_program)
            log_inputs = [x.output for x in self.recieve_reward_program.input_nodes]
            outputs = self.recieve_reward_program.run_presetinputs()
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve reward. Inputs: {log_inputs}. Outputs: {outputs}.")

            self.update_internal_state(listmult(outputs[2:2+self.internal_state_variable_count], outputs[1]))

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve reward, added own action controller to queue. ")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'neuron_action_controller'
                )
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron recieved reward, but cancelled due to neuron dying.")

    def run_move(self, timestep):
        if not self.dying:
            self.move_program.reset()
            self.set_global_pos(self.move_program, (0, 1, 2))
            self.set_internal_state_inputs(self.move_program)
            outputs = self.move_program.run_presetinputs()
            x_translation_pos = 1.0 if randcheck(outputs[0]) else 0.0
            y_translation_pos = 1.0 if randcheck(outputs[1]) else 0.0
            z_translation_pos = 1.0 if randcheck(outputs[2]) else 0.0
            x_translation_neg = 1.0 if randcheck(outputs[3]) else 0.0
            y_translation_neg = 1.0 if randcheck(outputs[4]) else 0.0
            z_translation_neg = 1.0 if randcheck(outputs[5]) else 0.0
            log_inputs = [x.output for x in self.move_program.input_nodes]
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran move. Inputs: {log_inputs}. Outputs: {outputs}.")

            if randcheck(outputs[6]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran move. Adding signal actions to queue.")
                self.addqueue_inner(
                    lambda: self.run_signal_axon(outputs[7:7+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_axon'
                )
                self.addqueue_inner(
                    lambda: self.run_signal_dendrite(outputs[7:7+self.signal_arity], timestep + 1), 
                    timestep + 1,
                    'neuron_signal_dendrite'
                )
            oldpos = [x for x in (self.x_glob, self.y_glob, self.z_glob)]
            self.x_glob += x_translation_pos - x_translation_neg
            self.y_glob += -y_translation_neg + y_translation_pos
            self.z_glob += -z_translation_neg + z_translation_pos
            self.x_loc += x_translation_pos - x_translation_neg
            self.y_loc += -y_translation_neg + y_translation_pos
            self.z_loc += -z_translation_neg + z_translation_pos
        
            self.grid, (self.x_loc, self.y_loc, self.z_loc), (self.x_glob, self.y_glob, self.z_glob) = \
                self.neuron_engine.update_neuron_position(
                    self, 
                    (self.x_loc, self.y_loc, self.z_loc),
                )
            for axon in self.axons:
                axon.update_pos(self.x_glob, self.y_glob, self.z_glob)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran move. {oldpos} -> [{self.x_glob}, {self.y_glob}, {self.z_glob}].")
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran move action, but cancelled due to neuron dying. ")

    def run_die(self, timestep):
        if not self.dying:
            self.die_program.reset()
            self.set_global_pos(self.die_program, (0, 1, 2))
            self.set_internal_state_inputs(self.die_program)
            outputs = self.die_program.run_presetinputs()
            log_inputs = [x.output for x in self.die_program.input_nodes]

            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran die. Inputs: {log_inputs}. Outputs: {outputs}.")
            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran die. Is dying.")
                self.dying = True
                self.neuron_engine.remove_neuron(self)
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran die. Killing child axons: {[x.id for x in self.axons]}. Killing child dendrites: {[x.id for x in self.dendrites]}")
                for axon in self.axons:
                    axon.die(timestep + 1)
                for dendrite in self.dendrites:
                    dendrite.die(timestep + 1)
            
                # send death signal optionally
                if randcheck(outputs[1]):
                    self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran die, calling run signal axon at this timestep. ")
                    # notice doing it at current timestep, breaking normal sequence, because dendrite.die is called at next timestep
                    self.addqueue_inner(
                        lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep, same_timestep = True), 
                        timestep,
                        'neuron_signal_axon'
                    )
                    self.addqueue_inner(
                        lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep, same_timestep = True), 
                        timestep,
                        'neuron_signal_dendrite'
                    )
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran die, but it is already dying.")

    def run_neuron_birth(self, timestep):
        if not self.dying:
            self.neuron_birth_program.reset()
            self.set_global_pos(self.neuron_birth_program, (0, 1, 2))
            self.set_internal_state_inputs(self.neuron_birth_program)
            outputs = self.neuron_birth_program.run_presetinputs()
            log_inputs = [x.output for x in self.neuron_birth_program.input_nodes]
            self.logger.log("engine_action", f"{self.id}, {timestep}: Ran neuron birth. Inputs: {log_inputs}. Outputs: {outputs}.")

            if randcheck(outputs[0]):
                new_neuron = Neuron(self.neuron_initialization_data,
                    self.axon_initialization_data,
                    self.neuron_engine,
                    self.x_glob,
                    self.y_glob,
                    self.z_glob,
                    self.x_loc,
                    self.y_loc,
                    self.z_loc,
                    self.signal_arity,
                    self.hox_variants,
                    self.counter,
                    self.grid,
                    logger = self.logger,
                    config = self.config)
                new_neuron.internal_states = outputs[1:1+new_neuron.internal_state_variable_count]
                self.neuron_engine.add_neuron_created(new_neuron)

                self.logger.log("engine_action", f"{self.id}, {timestep}: Ran neuron birth, gave birth to neuron {new_neuron.id}. Adding new neuron action controller to queue.")

                self.update_internal_state(
                    listmult(outputs[2+new_neuron.internal_state_variable_count:2+new_neuron.internal_state_variable_count+self.internal_state_variable_count],
                        outputs[1+new_neuron.internal_state_variable_count]))
                self.addqueue_inner(
                    lambda: new_neuron.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'neuron_neuron_birth'
                )
                own_dendrite = self.create_new_dendrite()
                target_axon = new_neuron.create_new_dendrite(False)
                own_dendrite.connect(target_axon)


        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran neuron birth, but cancelled due to neuron dying.")

        # Change state? Gen signals? 
    def run_action_controller(self, timestep):
        if not self.dying:
            self.action_controller_program.reset()
            self.set_global_pos(self.action_controller_program, (0, 1, 2))
            self.set_internal_state_inputs(self.action_controller_program)
            outputs = self.action_controller_program.run_presetinputs()
            log_inputs = [x.output for x in self.action_controller_program.input_nodes]

            for num in range(self.program_order_length):
                output = outputs[num]
                if randcheck(output):
                    func, name = self.program_order_lambda_factory(timestep + 1, num)
                    self.addqueue_inner(
                        func,
                        timestep + 1,
                        name
                    )
            
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran action controller. Inputs: {log_inputs}. Outputs: {outputs}.")
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran action controller, but cancelled due to neuron dying. ")

    # TODO: Axon missing hox selection - just use parent hox for simplicity for now
    def run_hox_selection(self):
        if not self.dying:
            self.hox_variant_selection_program.reset()
            self.set_global_pos(self.hox_variant_selection_program, (0, 1, 2))
            self.set_internal_state_inputs(self.hox_variant_selection_program)
            log_inputs = [x.output for x in self.hox_variant_selection_program.input_nodes]
            outputs = self.hox_variant_selection_program.run_presetinputs()
            chosen_variant = outputs.index(max(outputs))
            if self.selected_hox_variant != chosen_variant:
                self.neuron_engine.hox_switches_diagnostic += 1
            self.selected_hox_variant = chosen_variant
            self.load_hox_programs()
            self.logger.log("engine_action", f"{self.id}: Neuron hox selection, selected {self.selected_hox_variant}. Inputs: {log_inputs}. Outputs: {outputs}")
        else: 
            self.logger.log("engine_action", f"{self.id}: Neuron hox selection aborted, neuron dying")
    
    def __eq__(self, other):
        if type(other) is Neuron:
            return self.id == other.id
        return False


class Axon(CellObjectSuper):
    def __init__(self,
                 axon_initialization_data,
                 neuron_engine,
                 signal_arity,
                 counter,
                 neuron,
                 logger,
                 config,
                 seek_connection=True) -> None:
        super(Axon, self).__init__(logger, config)
        self.signal_arity = signal_arity
        self.config = config
        self.cgp_constant_count = len(self.config['cgp_function_constant_numbers'])
        
        self.id = counter.counterval()

        self.counter = counter


        self.program_order = [
            lambda t: self.run_break_connection(t),
            lambda t: self.run_die(t),
            lambda t: self.run_action_controller(t)
        ]
        self.program_order_length = 3

        self.internal_state_variable_count = axon_initialization_data['internal_state_variable_count']
        self.internal_states = []
        for _ in range(self.internal_state_variable_count):
            self.internal_states.append(0.0)

        self.neuron_engine = neuron_engine
        self.parent_x_glob = neuron.x_glob
        self.parent_y_glob = neuron.y_glob
        self.parent_z_glob = neuron.z_glob

        self.dying = False
        self.connected_dendrite = None
        self.neuron = neuron
        self.seek_dendrite_tries = self.config['seek_dendrite_tries']
        self.axon_initialization_data = axon_initialization_data

        self.recieve_signal_neuron_program = None
        self.recieve_signal_axon_program = None
        self.signal_dendrite_program =  None
        self.signal_neuron_program =  None
        self.accept_connection_program = None
        self.break_connection_program = None
        self.recieve_reward_program = None
        self.die_program = None
        self.action_controller_program = None

        self.load_hox_programs()
        self.neuron.grid.add_free_dendrite(self)
        if seek_connection:
            self.seek_dendrite_connection()

    def load_hox_programs(self):
        self.recieve_signal_neuron_program = self.axon_initialization_data['recieve_signal_neuron_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.recieve_signal_axon_program = self.axon_initialization_data['recieve_signal_dendrite_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.signal_dendrite_program =  self.axon_initialization_data['signal_dendrite_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.signal_neuron_program =  self.axon_initialization_data['signal_neuron_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.accept_connection_program = self.axon_initialization_data['accept_connection_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.break_connection_program = self.axon_initialization_data['break_connection_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.recieve_reward_program = self.axon_initialization_data['recieve_reward_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.die_program = self.axon_initialization_data['die_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program
        self.action_controller_program = self.axon_initialization_data['action_controller_programs']\
            .hex_variants[self.neuron.selected_hox_variant].program

    def connect(self, target_dendrite, timestep = None):
        if not type(target_dendrite) == OutputNeuron and not type(target_dendrite) == InputNeuron:
            if target_dendrite.neuron == self.neuron:
                warnings.warn("Tried to form auto-connection.")
                return False
            target_dendrite.connected_dendrite = self
            self.connected_dendrite = target_dendrite
            target_dendrite.neuron.grid.remove_free_dendrite(target_dendrite)
            if timestep is not None:
                self.logger.log("engine_action", f"{self.id}, {timestep}: Found connection to {self.connected_dendrite.id}. Type: Axon")
        else:
            target_dendrite.add_subscriber(self)
            self.connected_dendrite = target_dendrite
            if type(self.connected_dendrite) == InputNeuron:
                self.logger.log("engine_action", f"{self.id}, {timestep}: Found connection to {self.connected_dendrite.id}. Type: InputNeuron")
            elif type(self.connected_dendrite) == OutputNeuron:
                self.logger.log("engine_action", f"{self.id}, {timestep}: Found connection to {self.connected_dendrite.id}. Type: OutputNeuron")
        self.neuron.grid.remove_free_dendrite(self)
        return True

    def seek_dendrite_connection(self, timestep = None):
        self.neuron.neuron_engine.action_counts['dendrite_seek_connection'] += 1 
        if self.connected_dendrite is None:
            # Because seek_dendrite_connection can be added to the queue for the same axon-dendrite because a queued
            # action does not necessarily entile a future connection it is possible that this function is called when
            # an axon-dendrite already has a connection.
            dim_size = self.neuron_engine.get_size_in_neuron_positions_one_dim()
            max_x_dist = dim_size - self.parent_x_glob
            max_y_dist = dim_size - self.parent_y_glob
            max_z_dist = dim_size - self.parent_z_glob
            max_dist = max_x_dist + max_y_dist + max_z_dist # let's just do manhattan to start with because it is easier
            attempt = 0
            while attempt < self.seek_dendrite_tries:
                dist_target = int(math.floor((1-np.random.power(3))*max_dist))
                target_dendrite = self.neuron_engine.get_free_dendrite(self.neuron, dist_target, self in self.neuron.axons)
                if target_dendrite is None: 
                    break
                elif type(target_dendrite) is InputNeuron or type(target_dendrite) is OutputNeuron or (target_dendrite.neuron != self.neuron and type(target_dendrite) is Axon and target_dendrite.connected_dendrite is not None):
                    self.neuron.neuron_engine.action_counts['dendrite_accept_connection'] += 2
                    if target_dendrite.run_accept_connection(self, timestep) and \
                        self.run_accept_connection(target_dendrite, timestep):
                            return self.connect(target_dendrite, timestep)
                attempt += 1
            self.logger.log("engine_action", f"{self.id}, {timestep}: Failed to find connection.")
            self.neuron.grid.add_free_dendrite(self)


    def die(self, timestep):
        self.dying = True
        self.neuron.grid.remove_free_dendrite(self)
        if self.connected_dendrite is not None:
            if type(self.connected_dendrite) != OutputNeuron and type(self.connected_dendrite) != InputNeuron:
                self.connected_dendrite.neuron.grid.add_free_dendrite(self.connected_dendrite)
            self.connected_dendrite.remove_subscriber(self)
            self.connected_dendrite = None
        self.neuron.remove_dendrite(self)

    
    def addqueue_inner(self, lambdafunc, timestep, action_name):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id, action_name)


    def program_order_lambda_factory(self, timestep, index):
        if index == 0:
            return lambda: self.run_break_connection(timestep), "dendrite_break_connection"
        elif index == 1:
            return lambda: self.run_die(timestep), "dendrite_die"
        elif index == 2:
            return lambda: self.run_action_controller(timestep), "dendrite_action_controller"
        else: 
            raise Exception("Invalid index")


    def update_pos(self, x, y, z):
        self.parent_x_glob = x
        self.parent_y_glob = y
        self.parent_z_glob = z

    def remove_subscriber(self, subscriber):
        if self.connected_dendrite == subscriber:
            self.connected_dendrite = None

    def recieve_signal_setup(self, program, signals):
        program.reset()
        self.set_global_pos(program, (0, 1, 2))
        for num in range(3, 3+len(signals)):
            program.input_nodes[num].set_output(signals[num-3])
        self.set_internal_state_inputs(program)
        outputs = program.run_presetinputs()
        return outputs, [x.output for x in program.input_nodes]

    def run_recieve_signal_neuron(self, signals, timestep):
        if (not self.dying) and self.connected_dendrite is not None:
            outputs, log_inputs = self.recieve_signal_setup(self.recieve_signal_neuron_program, signals)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal neuron. Adding run signals to queue. Inputs: {log_inputs}. Outputs: {outputs}.")
            self.addqueue_inner(
                lambda: self.run_signal_dendrite(outputs[1:1+self.signal_arity], timestep + 1), 
                timestep + 1,
                'dendrite_signal_axon'
            )

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Neuron ran recieve signal neuron. Adding own action controller to queue. ")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'dendrite_action_controller'
                )

            self.update_internal_state(listmult(outputs[self.signal_arity+1:self.signal_arity+self.internal_state_variable_count+1], outputs[self.signal_arity+1+self.internal_state_variable_count]))

        elif self.connected_dendrite is None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Dendite ran recieve signal from neuron, but has no connected axon, seeking connection.")
            self.seek_dendrite_connection()
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Dendrite ran recieve signal from neuron, but cancelled due to dendrite dying.")
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if (not self.dying) and self.connected_dendrite is not None:
            outputs, log_inputs = self.recieve_signal_setup(self.recieve_signal_axon_program, signals)

            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran recieve signal from dendrite. Adding run signals to queue. Inputs: {log_inputs}. Outputs: {outputs}")
            self.addqueue_inner(
                lambda: self.run_signal_dendrite(outputs[1:1+self.signal_arity], timestep + 1), 
                timestep + 1,
                'axon_signal_dendrite'
            )

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran recieve signal from dendrite, added action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'dendrite_action_controller'
                )
            
            self.update_internal_state(listmult(outputs[self.signal_arity+1:self.signal_arity+self.internal_state_variable_count+1], outputs[self.signal_arity+1+self.internal_state_variable_count]))
        
        elif self.connected_dendrite is None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran recieve signal from dendrite, but has no connected dendrite. Seeking for free dendrite.")
            self.seek_dendrite_connection()
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran recieve signal from dendrite, but is dying.")

    
    def run_recieve_signal_axon(self, signals, timestep):
        if (not self.dying) and self.connected_dendrite is not None:
            outputs, log_inputs = self.recieve_signal_setup(self.recieve_signal_axon_program, signals)

            self.logger.log("engine_action", f"{self.id}, {timestep}: Dendrite ran recieve signal from axon. Adding run signals to queue. Inputs: {log_inputs}. Outputs: {outputs}.")

            self.addqueue_inner(
                lambda: self.run_signal_neuron(outputs[1:1+self.signal_arity], timestep + 1), 
                timestep + 1,
                'dendrite_signal_neuron'
            )

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Dendrite ran recieve signal from axon. Adding own action controller to queue. ")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'dendrite_action_controller'
                )
            
            self.update_internal_state(listmult(outputs[self.signal_arity+1:self.signal_arity+self.internal_state_variable_count+1], outputs[self.signal_arity+1+self.internal_state_variable_count]))
        
        elif self.connected_dendrite is None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Dendrite ran recieve signal from axon, but has no connected axon. Seeking for connection.")
            self.seek_dendrite_connection()
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Dendirte ran recieve signal from axon, but is dying.")

    
    def send_signal_setup(self, program, signals):
        program.reset()
        self.set_global_pos(program, (0, 1, 2))
        self.set_internal_state_inputs(program)
        for num in range(self.signal_arity):
            program.input_nodes[num+3+self.internal_state_variable_count+self.cgp_constant_count].set_output(signals[num])
        return program.run_presetinputs(), [x.output for x in program.input_nodes]

    def run_signal_dendrite(self, signals, timestep):
        if (not self.dying) and self.connected_dendrite is not None: 
            outputs, log_inputs = self.send_signal_setup(self.signal_dendrite_program, signals)

            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran signal dendrite. Inputs: {log_inputs}. Outputs: {outputs}.")

            if randcheck(outputs[0]):
                if type(self.connected_dendrite) != InputNeuron:
                    self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran signal dendrite. Adding recieve signal in connected axon to queue.")
                    # TODO adding sending signals on dying does not work at all - old note, unsure about current status
                    dendrite = self.connected_dendrite
                    def _internal_handler(self, outputs, timestep, dendrite):
                        if dendrite is not None and dendrite != InputNeuron:
                            dendrite.run_recieve_signal_axon(
                                outputs[1:1+self.signal_arity],
                                timestep + 1
                            )
                    self.addqueue_inner(
                        lambda: _internal_handler(self, outputs, timestep, dendrite),
                        timestep + 1,
                        'axon_recieve_signal_dendrite'
                    )
                else:
                    self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran signal dendrite, but is connected to InputNeuron: No action.")


            self.update_internal_state(listmult(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count], outputs[1+self.signal_arity]))

            if randcheck(outputs[-1] >= 1.0):
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1),
                    timestep + 1,
                    'dendrite_action_controller'
                )
            
        if self.connected_dendrite is None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran signal dendrite, but has no connected dendrite. Seeking connection.")
            self.seek_dendrite_connection(timestep)
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon ran signal dendrite, but is dying.")

    def run_signal_neuron(self, signals, timestep):
        if not self.dying: 
            outputs, log_inputs = self.send_signal_setup(self.signal_neuron_program, signals)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran signal neuron. Inputs: {log_inputs}. Outputs: {outputs}.")

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran signal neuron. Added recieving signal in neuron to queue.")
                neuron = self.neuron
                # In general it is important to refering to specific objects instead of class properties when adding lambda's to
                # the queue when communicating between CellObjects - otherwise the class attribute may refer to the wrong entity
                # upon execution, breaking the temporal sequencing. (Mostly a concern between axon-dendrite outgoing connections
                # to other axon-dendrites, inputneurons or outputneurons)
                self.addqueue_inner(
                    lambda: neuron.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1,
                    'neuron_recieve_axon_signal'
                )

            self.update_internal_state(listmult(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count], outputs[1+self.signal_arity]))

            if randcheck(outputs[-1]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran signal neuron. Adding own action controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep+1),
                    timestep + 1,
                    'dendrite_action_controller'
                )
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran signal neuron, but is dying.")

    
        
    def run_accept_connection(self, dendrite, timestep = None):
        if (not self.dying) and self.connected_dendrite is None:
            self.accept_connection_program.reset()
            if type(dendrite) == Axon:
                connection_target_type = "Axon"
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [dendrite.neuron.x_glob, dendrite.neuron.y_glob, dendrite.neuron.z_glob] + \
                    dendrite.internal_states + self.config['cgp_function_constant_numbers']
            elif type(dendrite) == InputNeuron or OutputNeuron:
                if type(dendrite) == InputNeuron:
                    connection_target_type = "InputNeuron"
                else:
                    connection_target_type = "OutputNeuron"
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [dendrite.x, dendrite.y, dendrite.z] + \
                    [0 for x in range(len(self.internal_states))] + self.config['cgp_function_constant_numbers']
            outputs = self.accept_connection_program.run(program_inputs)

            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran accept connection. Connection target type: {connection_target_type}. Inputs: {program_inputs}. Outputs: {outputs}.")
            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran accept connection, accepted connection.")
                return True

            self.update_internal_state(listmult(outputs[2:2+self.internal_state_variable_count], outputs[1]))

            return False
        elif self.connected_dendrite is not None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran accept connection, but already has a connection. Ignored action.")
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran accept connection, but is dying.")
        return False # shouldn't happen maybe check for this TODO indicates fault in seek_dendrite_connection code
    
    def run_break_connection(self, timestep = None): 
        if (not self.dying) and self.connected_dendrite is not None: 
            self.break_connection_program.reset()
            if type(self.connected_dendrite) == Axon:
                connection_target_type = "Axon"
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [self.connected_dendrite.parent_x_glob, 
                        self.connected_dendrite.parent_y_glob, 
                        self.connected_dendrite.parent_z_glob
                    ] + \
                    self.connected_dendrite.internal_states + self.config['cgp_function_constant_numbers']
            else:
                if type(self.connected_dendrite) == InputNeuron:
                    connection_target_type = "InputNeuron"
                else:
                    connection_target_type = "OutputNeuron"
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [self.connected_dendrite.x, 
                        self.connected_dendrite.y, 
                        self.connected_dendrite.z
                    ] + \
                    [0 for x in range(len(self.internal_states))] + self.config['cgp_function_constant_numbers']
            outputs = self.break_connection_program.run(program_inputs)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran break connection. Connection target: {connection_target_type}. Inputs: {program_inputs}. Outputs: {outputs}.")
            

            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran break connection. Seeking new connection.")
                self.connected_dendrite.remove_subscriber(self)
                if type(self.connected_dendrite) == Axon:
                    if not self.connected_dendrite.dying: 
                        self.connected_dendrite.neuron.grid.add_free_dendrite(self.connected_dendrite)
                        if timestep is not None:
                            dendrite = self.connected_dendrite
                            self.addqueue_inner(
                                lambda: dendrite.seek_dendrite_connection(), 
                                timestep + 1,
                                'dendrite_seek_connection'
                            )
                        else:
                            self.connected_dendrite.seek_dendrite_connection()
                if not self.dying: # this can never happen though?
                    self.neuron.grid.add_free_dendrite(self)
                    if timestep is not None: 
                        self.addqueue_inner(
                            lambda: self.seek_dendrite_connection(),
                            timestep + 1,
                            'dendrite_seek_connection'
                        )
                    else: 
                        self.seek_dendrite_connection()
        elif self.connected_dendrite is None:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran break connection, but is not connected to anything.")
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran break connection, but is dying.")

    def run_recieve_reward(self, reward, timestep): 
        if not self.dying: 
            self.recieve_reward_program.reset()
            program_inputs = [reward] + [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states + self.config['cgp_function_constant_numbers']
            outputs = self.recieve_reward_program.run(program_inputs)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran recieve reward. Inputs: {program_inputs}. Outputs: {outputs}.")
            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran recieve reward. Added aciton-controller to queue.")
                self.addqueue_inner(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    'dendrite_action_controller'
                )

            self.update_internal_state(listmult(outputs[2:2+self.internal_state_variable_count], outputs[1]))
        else:
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon dendrite ran recieve reward, but is dying.")
    
    def run_die(self, timestep):
        if not self.dying: 
            self.die_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states + self.config['cgp_function_constant_numbers']
            outputs = self.die_program.run(program_inputs)
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran die. Inputs: {program_inputs}. Outputs: {outputs}.")
            if randcheck(outputs[0]):
                self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran die, now dies. Adding run signals to queue..")
                self.dying = True
                self.addqueue_inner(
                    lambda: self.run_signal_dendrite(outputs[1:1+self.signal_arity], timestep),
                    timestep,
                    'dendrite_axon_death_connection_signal'
                )
                self.addqueue_inner(
                    lambda: self.run_signal_neuron(outputs[1:1+self.signal_arity], timestep),
                    timestep,
                    'dendrite_axon_death_neuron_signal'
                )
                self.addqueue_inner(
                    lambda: self.die(timestep + 1), 
                    timestep + 1,
                    'dendrite_die'
                )
        else: 
            self.logger.log("engine_action", f"{self.id}, {timestep}: Axon-dendrite ran die, but already dying.")

    def run_action_controller(self, timestep):
        self.action_controller_program.reset()
        self.set_global_pos(self.action_controller_program, (0, 1, 2))
        self.set_internal_state_inputs(self.action_controller_program)
        outputs = self.action_controller_program.run_presetinputs()
        log_inputs = [x.output for x in self.action_controller_program.input_nodes]
        self.logger.log("engine_action", f"{self.id}, {timestep}: Ran action controller. Inputs: {log_inputs}. Outputs: {outputs}.")
        for num in range(self.program_order_length):
            output = outputs[num]
            if output >= 1.0:
                action, name = self.program_order_lambda_factory(timestep + 1, num) 
                # For some reason action is sometimes a lambda func inside a tuple. I have no idea why.
                if type(action) == tuple:
                    action = action[0]
                self.addqueue_inner(
                    action,
                    timestep + 1,
                    name
                )

    def __eq__(self, other):
        if type(other) is Axon:
            return self.id == other.id
        return False