# here define the world space
from HelperClasses import randchoice, randcheck
import math
import numpy as np

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
        self.changed = False
        self.not_changed_count = 0

        self.input_arity = input_arity
        self.output_arity = output_arity
        self.grid_count = grid_count  # Grids in grid grid per dimension
        self.grid_size = grid_size   # neurons per grid per dimension
        self.actions_max = actions_max
        self.actions_count = 0  # For counting how many actions engine is allowed to take
        self.action_index = 0  # Pointing to next action in queue - different, as elements in queue can be replaced with "skip" token
        self.input_neurons = []
        self.middle_grid_input = math.floor(grid_count/2)
        self.middle_grid_output = math.floor(grid_count/2+1)
        self.init_grids()
        if input_arity > grid_size:
            raise Exception("grid size too small to contain inputs")
        if output_arity > grid_size: 
            raise Exception("grid size too small to contain outputs")
        self.neurons = []
        self.init_neurons()
        self.action_queue = []
        self.timestep_indexes = []
        self.debugging = debugging
        self.timestep = 0 
    
    def get_size_in_neuron_positions_one_dim(self):
        return self.grid_count * self.grid_size

    def get_current_timestep(self):
        return self.timestep

    def get_free_dendrite(self, neuron, target_distance):
        _target_dist = target_distance // self.grid_size
        if len(self.free_connection_grids) != 0:
            if _target_dist == 0 and neuron.grid in self.free_connection_grids:
                _x = [x for x in neuron.grid.free_dendrites if x not in neuron.dendrites and x not in neuron.axons]
                if len(_x) != 0:
                    return randchoice(_x)
                else:
                    input_and_output_neurons = [x for x in neuron.grid.neurons if type(x) == InputNeuron or type(x) == OutputNeuron]
                    return randchoice(input_and_output_neurons)
            else: 
                dists = [(x, abs(self.approximate_distance_grid(neuron.grid, x) - _target_dist)) for x in self.free_connection_grids]
                best_match_grid = min(dists, key=lambda x: x[1])[0]
                free_dendrites = best_match_grid.free_dendrites
                if not len(free_dendrites) == 0:
                    return randchoice(best_match_grid.free_dendrites)
                else:
                    input_and_output_neurons = [x for x in best_match_grid.neurons if type(x) == InputNeuron or type(x) == OutputNeuron]
                    return randchoice(input_and_output_neurons)
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

        if globalx >= self.grid_count*self.grid_size or globaly >= self.grid_count*self.grid_size or globalz >= self.grid_count*self.grid_size:
            raise Exception("Global position out of bounds")
        return self.grids[gridx][gridy][gridz], (localx, localy, localz), (globalx, globaly, globalz)


    def init_grids(self):
        self.grids = [[[None for _ in range(self.grid_count)] for _ in range(self.grid_count)] for _ in range(self.grid_count)]
        for n1 in range(self.grid_count):
            for n2 in range(self.grid_count):
                for n3 in range(self.grid_count):
                    self.grids[n1][n2][n3] = Grid(n1, n2, n3, self.grid_size, self)
        self.free_connection_grids = [
            self.grids[self.middle_grid_input][self.middle_grid_input][self.middle_grid_input],
            self.grids[self.middle_grid_output][self.middle_grid_output][self.middle_grid_output]
        ]

    
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
                self.middle_grid_input*self.grid_size + num))
        self.output_neurons = []
        for _ in range(self.output_arity):
            self.output_neurons.append(OutputNeuron(
                self.grids[self.middle_grid_output][self.middle_grid_output][self.middle_grid_output],
                self.middle_grid_output*self.grid_size + num,
                self.middle_grid_output*self.grid_size + num,
                self.middle_grid_output*self.grid_size + num))

        
    def reset(self):
        self.actions_count = 0
        self.action_index = 0
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []
        self.grids = []
        self.init_grids()
        self.init_neurons()
        self.timestep = 0

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
        for num in range(len(self.input_neurons)):
            self.input_neurons[num].value = input_sample[num]
        self.action_queue = []
        self.timestep_indexes = []
        for input_neuron in self.input_neurons:
            for axon in input_neuron.subscribers:
                self.add_action_to_queue(
                    lambda: axon.run_recieve_signal_dendrite(
                        [input_neuron.value for _ in range(self.signal_arity)],
                        0
                    ),
                    0,
                    axon.id
                )
        self.timestep = 0
        self.actions_count = 0
        self.action_index = 0
        for neuron in self.neurons:
            self.add_action_to_queue(
                lambda: neuron.run_action_controller(0),
                0,
                neuron.id
            )
        while len(self.action_queue) > self.action_index and self.actions_count < self.actions_max:
            timestep, action, _ = self.action_queue[self.action_index]
            if not action == "skip":
                action()  # runs action
                self.timestep = timestep
                self.actions_count += 1
            self.action_index += 1
        outputs = [output_neuron.value for output_neuron in self.output_neurons]
        return outputs

    def run(self, problem):
        # for every sample in inputs run it, some way of evaluating result as well
        # run run_sample for each sample some amount of times
        # then use eval function of problem class to give reward
        # Assumes goal is to minimize error
        self.reset()
        cumulative_error = 0
        exec_instances = 0
        for num in range(self.instances_per_iteration):
            exec_instances += 1
            self.changed = False
            instance = problem.get_problem_instance()
            result = self.run_sample(instance)
            error = problem.error(instance, result)
            if len(self.neurons) == 0:
              error += 2000
            no_input_connections = True
            # Always bad
            for node in self.input_neurons:
                if len(node.subscribers) != 0:
                    no_input_connections = False
                    break
            if no_input_connections:
                error += 2000
            
            cumulative_error += error
            reward = problem.get_reward(error)
            self.action_queue = []
            self.timestep_indexes = []
            self.timestep = 0
            self.actions_count = 0
            self.action_index = 0
            for neuron in self.neurons:
                self.add_action_to_queue(
                    lambda: neuron.run_recieve_reward(reward, 0),
                    0,
                    neuron.id
                )
                for axon in neuron.dendrites + neuron.axons:
                    self.add_action_to_queue(
                        lambda: axon.run_recieve_reward(reward, 0),
                        0,
                        axon.id
                    )
            while len(self.action_queue) > self.action_index and self.actions_count < self.actions_max:
                timestep, action, _ = self.action_queue[self.action_index]
                if not action == "skip":
                    action()  # runs action
                    self.timestep = timestep
                    self.actions_count += 1
                self.action_index += 1
            if not self.changed:
                if self.not_changed_count > 10:
                    break
                else:
                    self.not_changed_count += 1
            else:
                self.not_changed_count = 0
        return cumulative_error/exec_instances

    def add_action_to_queue(self, action, timestep, id):
        if action is None:
            raise Exception("None action added")
        elif type(action) is tuple:
            raise Exception("Tuple added")
        timesteps = [x[0] for x in self.timestep_indexes]
        timestep_relative_order = 0
        for x in timesteps:
            if timestep > x: 
                timestep_relative_order += 1
        if timestep_relative_order >= len(self.timestep_indexes):
            self.action_queue.append([timestep, action, id])
            self.timestep_indexes.append([timestep, len(self.action_queue)-1, id])
        else:
            pos = self.timestep_indexes[timestep_relative_order][1]+1
            self.action_queue.insert(pos, [timestep, action, id])
            self.timestep_indexes.insert(timestep_relative_order, [timestep, pos, id])
            for x in self.timestep_indexes[timestep_relative_order+1:]:
                x[1] = x[1]+1

    def remove_actions_id(self, id):
        to_remove_indexes = []
        for num in range(len(self.timestep_indexes)):
            _, _, _id = self.timestep_indexes[num]
            if id == _id:
                to_remove_indexes.append(num)
        to_remove_indexes.sort(reverse=True)
        actions_indexes_to_remove = []
        for val in to_remove_indexes:
            _, action_index, _ = self.timestep_indexes[val]
            actions_indexes_to_remove.append(action_index)
        for val in to_remove_indexes:
            self.timestep_indexes.remove(self.timestep_indexes[val])
        actions_indexes_to_remove.sort(reverse=True)
        for val in actions_indexes_to_remove:
            self.action_queue[val] = (self.action_queue[val][0], "skip", self.action_queue[val][1])  # A bit hacky maybe, but far easier than recalculating indexes in timestep indexes

    def remove_neuron(self, neuron):
        self.changed = True
        nid = neuron.id
        neuron.grid.remove_neuron(neuron)
        self.remove_actions_id(nid)

    def add_neuron(self, neuron_pos, neuron_internal_state):
        self.changed = True
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
        new_neuron.internal_states = neuron_internal_state
        grid.add_neuron(new_neuron)
        self.neurons.append(new_neuron)

    def add_neuron_created(self, neuron):
        self.changed = True
        neuron_pos = (neuron.x_glob, neuron.y_glob, neuron.z_glob)
        grid = self.grids[int(neuron_pos[0]//self.grid_size)][int(neuron_pos[1]//self.grid_size)][int(neuron_pos[2]//self.grid_size)]
        neuron.grid = grid
        grid.add_neuron(neuron)
        self.neurons.append(neuron)
    
    def remove_dendrite(self, dendrite):
        self.changed = True
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
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in
        self.grid = grid
        grid.add_neuron(self)
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

class OutputNeuron():
    def __init__(self, grid, x, y, z) -> None:
        # should know which grid it is in 
        self.grid = grid
        grid.add_neuron(self)
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
        print("Output recieve signal")
        self.value = signals[0] 



def addqueue(neuron_engine, lambdafunc, timestep, id):
    neuron_engine.add_action_to_queue(
        lambdafunc,
        timestep,
        id
    )


class CellObjectSuper:
    def __init__(self):
        self.x_glob = None
        self.y_glob = None
        self.z_glob = None

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
    
    def update_internal_state(self, deltas):
        for num in range(self.internal_state_variable_count):
            self.internal_states[num] += deltas[num]






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
                grid) -> None:
        super(CellObjectSuper, self).__init__()
        self.counter = counter
        self.id = self.counter.counterval()
        self.neuron_initialization_data = neuron_initialization_data
        self.axon_initialization_data = axon_initialization_data

        self.axon_birth_program = self.neuron_initialization_data['axon_birth_program']
        self.signal_axon_program = self.neuron_initialization_data['signal_axon_program']
        self.recieve_axon_signal_program = self.neuron_initialization_data['recieve_axon_signal_program']
        self.recieve_reward_program = self.neuron_initialization_data['recieve_reward_program']
        self.move_program = self.neuron_initialization_data['move_program']
        self.die_program = self.neuron_initialization_data['die_program']
        self.neuron_birth_program = self.neuron_initialization_data['neuron_birth_program']
        self.action_controller_program = self.neuron_initialization_data['action_controller_program']
        self.hox_variant_selection_program = self.neuron_initialization_data['hox_variant_selection_program']
        self.internal_state_variable_count = self.neuron_initialization_data['internal_state_variable_count']
        self.axons = []
        self.dendrites = []

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
        self.selected_hox_variant = None
        self.run_hox_selection()
        self.grid = grid
        self.program_order_length = 7

    def addqueue(self, lambdafunc, timestep):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id)

    def run_hox_selection(self):
        if not self.dying:
            self.hox_variant_selection_program.reset()
            self.set_global_pos(self.hox_variant_selection_program, (0, 1, 2))
            self.set_internal_state_inputs(self.hox_variant_selection_program)
            outputs = self.hox_variant_selection_program.run_presetinputs()
            chosen_variant = outputs.index(max(outputs))
            self.selected_hox_variant = chosen_variant


    def program_order_lambda_factory(self, timestep, index):
        if index == 0:
            return lambda: self.run_dendrite_birth(timestep)
        elif index == 1:
            return lambda: self.run_axon_birth(timestep)
        elif index == 2:
            return lambda: self.run_move(timestep)
        elif index == 3:
            return lambda: self.run_die(timestep)
        elif index == 4:
            return lambda: self.run_neuron_birth(timestep)
        elif index == 5:
            return lambda: self.run_hox_selection()
        elif index == 6:
            return lambda: self.run_action_controller(timestep)
        else:
            raise Exception("Invalid program order index")


    def run_dendrite_birth(self, timestep):
        if not self.dying:
            # TODO REALLY missing something in input, none in output
            self.axon_birth_program.reset()
            self.set_global_pos(self.axon_birth_program, (0, 1, 2))
            self.set_internal_state_inputs(self.axon_birth_program)
            self.axon_birth_program.input_nodes[6].set_output(len(self.dendrites))
            outputs = self.axon_birth_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if randcheck(outputs[0]):
                self.dendrites.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self))
                # TODO sometimes gives list index out of range
                dendrite = self.dendrites[-1]
                self.addqueue(
                    lambda: dendrite.run_action_controller(timestep + 1),
                    timestep + 1
                )
                
            if randcheck(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
            
            if randcheck(outputs[2+self.signal_arity]):
                self.update_internal_state(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count])
            
            if randcheck(outputs[-1]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep+1),
                    timestep + 1
                )

     
    def run_axon_birth(self, timestep):
        if not self.dying:
            self.axon_birth_program.reset()
            self.set_global_pos(self.axon_birth_program, (0, 1, 2))
            self.axon_birth_program.input_nodes[3].set_output(len(self.axons))
            self.set_internal_state_inputs(self.axon_birth_program)
            outputs = self.axon_birth_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if randcheck(outputs[0]):
                self.axons.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self))
                axon = self.axons[-1]
                self.addqueue(
                    lambda: axon.run_action_controller(timestep+1), 
                    timestep + 1
                )
            
            if randcheck(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
            
            if randcheck([2+self.signal_arity]):
                self.update_internal_state(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count])
            
            if randcheck([-1]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1),
                    timestep + 1
                )
    
    def set_signal_inputs(self, program, signals):
        if len(signals) != self.signal_arity:
            raise Exception("Invalid signal dimensionality")
        for num in range(len(signals)):
            program.input_nodes[num].set_output(signals[num])

    def run_signal_dendrite(self, signals, timestep):
        self.signal_axon_program.reset()
        self.set_signal_inputs(self.signal_axon_program, signals)
        self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals)+3))
        self.set_internal_state_inputs(self.signal_axon_program)
        outputs = self.signal_axon_program.run_presetinputs()
        if randcheck(outputs[0]):
            for dendrite in self.dendrites:
                self.addqueue(
                    lambda: dendrite.run_recieve_signal_neuron(
                        outputs[1:1+self.signal_arity], 
                        timestep + 1
                    ), 
                    timestep + 1
                )
        
        if randcheck(outputs[1+self.signal_arity]):
            self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
        
        if randcheck(outputs[-1]):
            self.addqueue(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1
            )
    

    def run_signal_axon(self, signals, timestep):
        self.signal_axon_program.reset()
        self.set_signal_inputs(self.signal_axon_program, signals)
        self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals)+3))
        self.set_internal_state_inputs(self.signal_axon_program)
        outputs = self.signal_axon_program.run_presetinputs()

        if randcheck(outputs[0]):
            for axon in self.axons:
                self.addqueue(
                    lambda: axon.run_recieve_signal_neuron(
                        outputs[1 + self.internal_state_variable_count:1+self.internal_state_variable_count+self.signal_arity], 
                        timestep + 1
                    ), 
                    timestep + 1
                )
        
        if randcheck(outputs[1+self.signal_arity]):
            self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
        
        if randcheck(outputs[-1]):
            self.addqueue(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1
            )

    
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying: 
            self.recieve_axon_signal_program.reset()
            self.set_signal_inputs(self.recieve_axon_signal_program, signals)
            self.set_global_pos(self.recieve_axon_signal_program, range(len(signals), len(signals)+3))
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            if randcheck(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:self.signal_arity+3], timestep + 1), 
                    timestep + 1
                )

            if randcheck(outputs[2]):
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[3:self.signal_arity+3], timestep + 1), 
                    timestep + 1
                )


            if randcheck(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            
            if randcheck(outputs[3+self.signal_arity]):
                self.update_internal_state(outputs[3+self.signal_arity:3+self.internal_state_variable_count+self.signal_arity])
    
    def run_recieve_signal_axon(self, signals, timestep):
        if not self.dying:
            self.recieve_axon_signal_program.reset()
            self.set_signal_inputs(self.recieve_axon_signal_program, signals)
            self.set_global_pos(self.recieve_axon_signal_program, range(len(signals), len(signals)+3))
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            self.update_internal_state(outputs[3:3+self.internal_state_variable_count])

            if randcheck(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if randcheck(outputs[2]):
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )


            if randcheck(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
        
    def run_recieve_reward(self, reward, timestep):
        if not self.dying:
            self.recieve_reward_program.reset()
            self.recieve_reward_program.input_nodes[0].set_output(reward)
            self.set_global_pos(self.recieve_reward_program, (1, 2, 3))
            self.set_internal_state_inputs(self.recieve_reward_program)
            outputs = self.recieve_reward_program.run_presetinputs()

            if randcheck(outputs[1]):
                self.update_internal_state(outputs[2:2+self.internal_state_variable_count])

            if randcheck(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )

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

            if randcheck(outputs[6]):
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[7:7+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[7:7+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
            
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
    
    def run_die(self, timestep):
        if not self.dying:
            self.die_program.reset()
            self.set_global_pos(self.die_program, (0, 1, 2))
            self.set_internal_state_inputs(self.die_program)
            outputs = self.die_program.run_presetinputs()
            if randcheck(outputs[0]):
                self.dying = True
                self.neuron_engine.remove_neuron(self)
                for axon in self.axons:
                    axon.die(timestep + 1)
                for dendrite in self.dendrites:
                    dendrite.die(timestep + 1)
            
                # send death signal optionally
                if randcheck(outputs[1]):
                    # notice doing it at current timestep, breaking normal sequence, because dendrite.die is called at next timestep
                    self.addqueue(
                        lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep), 
                        timestep
                    )
                    self.addqueue(
                        lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep), 
                        timestep
                    )
    
    def run_neuron_birth(self, timestep):
        if not self.dying:
            self.neuron_birth_program.reset()
            self.set_global_pos(self.neuron_birth_program, (0, 1, 2))
            self.set_internal_state_inputs(self.neuron_birth_program)
            outputs = self.neuron_birth_program.run_presetinputs()
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
                    self.grid)
                new_neuron.internal_states = outputs[1:1+new_neuron.internal_state_variable_count]
                self.neuron_engine.add_neuron_created(new_neuron)

                if randcheck(outputs[1+new_neuron.internal_state_variable_count]):
                    self.update_internal_state(
                        outputs[2+new_neuron.internal_state_variable_count:2+new_neuron.internal_state_variable_count+self.internal_state_variable_count])
                self.addqueue(
                    lambda: new_neuron.run_action_controller(timestep + 1), 
                    timestep + 1
                )
    
    # Change state? Gen signals? 
    def run_action_controller(self, timestep):
        if not self.dying:
            self.action_controller_program.reset()
            self.set_global_pos(self.action_controller_program, (0, 1, 2))
            self.set_internal_state_inputs(self.action_controller_program)
            outputs = self.action_controller_program.run_presetinputs()
            for num in range(self.program_order_length):
                output = outputs[num]
                if randcheck(output):
                    self.addqueue(
                        self.program_order_lambda_factory(timestep + 1, num),
                        timestep + 1
                    )
    
    def execute_action_controller_output(self, outputs, timestep):
        for num in range(len(outputs)):
            output = outputs[num]
            if randcheck(output):
                self.add_to_queue(num, timestep + 1)
    
    def remove_dendrite(self, dendrite):
        if dendrite in self.axons:
            self.axons.remove(dendrite)
        if dendrite in self.dendrites:
            self.dendrites.remove(dendrite)
        self.neuron_engine.remove_dendrite(dendrite)

class Axon(CellObjectSuper):
    def __init__(self,
                 axon_initialization_data,
                 neuron_engine,
                 signal_arity,
                 counter,
                 neuron) -> None:
        super(CellObjectSuper, self).__init__()
        self.signal_arity = signal_arity
        self.counter = counter
        self.id = self.counter.counterval()

        self.recieve_signal_neuron_program = axon_initialization_data['recieve_signal_neuron_program']
        self.recieve_signal_axon_program = axon_initialization_data['recieve_signal_dendrite_program']
        self.signal_dendrite_program =  axon_initialization_data['signal_dendrite_program']
        self.signal_neuron_program =  axon_initialization_data['signal_neuron_program']
        self.accept_connection_program = axon_initialization_data['accept_connection_program']
        self.break_connection_program = axon_initialization_data['break_connection_program']
        self.recieve_reward_program = axon_initialization_data['recieve_reward_program']
        self.die_program = axon_initialization_data['die_program']
        self.action_controller_program = axon_initialization_data['action_controller_program']

        self.program_order = [
            lambda t: self.run_break_connection(t),
            lambda t: self.run_die(t),
            lambda t: self.run_action_controller(t)
        ]
        self.program_order_length = 3

        # TODO REALLY missing internal_state_variable_count key
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
        self.seek_dendrite_tries = 10


        self.neuron.grid.add_free_dendrite(self)
        self.seek_dendrite_connection()


    
    def addqueue(self, lambdafunc, timestep):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id)


    def program_order_lambda_factory(self, timestep, index):
        if index == 0:
            return lambda: self.run_break_connection(timestep),
        elif index == 1:
            return lambda: self.run_die(timestep),
        elif index == 2:
            return lambda: self.run_action_controller(timestep)
        else: 
            raise Exception("Invalid index")


    def update_pos(self, x, y, z):
        self.parent_x_glob = x
        self.parent_y_glob = y
        self.parent_z_glob = z

    # TODO when run break connection? Think maybe run action controller a lot

    def recieve_signal_setup(self, program, signals):
        program.reset()
        for num in range(0, len(signals)):
            program.input_nodes[num].set_output(signals[num])
        self.set_global_pos(program, range(len(signals), len(signals)+3))
        self.set_internal_state_inputs(program)
        outputs = program.run_presetinputs()
        return outputs

    def run_recieve_signal_neuron(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None:
            outputs = self.recieve_signal_setup(self.recieve_signal_neuron_program, signals)
            if randcheck(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if randchoice(outputs[2]):
                self.addqueue(
                    lambda: self.run_signal_neuron(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if randchoice(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            if randchoice(outputs[self.signal_arity+3]):
                self.update_internal_state(outputs[self.signal_arity+4:self.signal_arity+self.internal_state_variable_count+4])

        elif self.connected_dendrite is None:
            self.seek_dendrite_connection()
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None:
            outputs = self.recieve_signal_setup(self.recieve_signal_axon_program, signals)
        
            if randchoice(outputs[1]):
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if randchoice(outputs[2]):
                self.addqueue(
                    lambda: self.run_signal_neuron(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if randchoice(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            
            if randchoice(outputs[self.signal_arity+3]):
                self.update_internal_state(outputs[self.signal_arity+4:self.signal_arity+self.internal_state_variable_count+4])
        
        elif self.connected_dendrite is None:
            self.seek_dendrite_connection()
    
    def send_signal_setup(self, program, signals):
        program.reset()
        self.set_global_pos(program, (0, 1, 2))
        self.set_internal_state_inputs(program)
        for num in range(self.signal_arity):
            program.input_nodes[num+6].set_output(signals[num])
        return program.run_presetinputs()

    def run_signal_neuron(self, signals, timestep):
        if not self.dying: 
            outputs = self.send_signal_setup(self.signal_neuron_program, signals)
            if randchoice(outputs[0]):
                self.addqueue(
                    lambda: self.neuron.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1
                )
            if randchoice(outputs[1+self.signal_arity]):
                self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
            if randchoice(outputs[-1]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep+1),
                    timestep + 1
                )
# TODO: Remove all checks for output like >= 1.0 as they do not give a smooth fitness landscape. 
# Instead do scaling up to 1 for internal state adjustment, 
# probability for action controller
# Always send signals and such when doing it directly and not through action controller
# For action controller do probability scaling up to 1
    def run_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None: 
            outputs = self.send_signal_setup(self.signal_dendrite_program, signals)
            if randchoice(outputs[0]):
                if type(self.connected_dendrite) is Neuron:
                    self.addqueue(
                        lambda: self.connected_dendrite.run_recieve_signal_axon(
                            outputs[1:1+self.signal_arity],
                            timestep + 1),
                        timestep + 1
                    )
                # elif output TODO
            if randchoice(outputs[1+self.signal_arity] >= 1.0):
                self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
            if randchoice(outputs[-1] >= 1.0):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1),
                    timestep + 1
                )
            
        if self.connected_dendrite is None:
            self.seek_dendrite_connection(timestep)
        
    def run_accept_connection(self, dendrite, timestep = None):
        if not self.dying and self.connected_dendrite is None:
            self.accept_connection_program.reset()
            if type(dendrite) == Axon:
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [dendrite.neuron.x_glob, dendrite.neuron.y_glob, dendrite.neuron.z_glob] + \
                    dendrite.internal_states
            elif type(dendrite) == InputNeuron or OutputNeuron:
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [dendrite.x, dendrite.y, dendrite.z] + \
                    [0 for x in range(len(self.internal_states))]
            outputs = self.accept_connection_program.run(program_inputs)
            if randchoice(outputs[0]):
                return True
            if randchoice(outputs[1]):
                self.update_internal_state(outputs[2:2+self.internal_state_variable_count])
            return False
        return False # shouldn't happen maybe check for this TODO indicates fault in seek_dendrite_connection code
    
    def run_break_connection(self, timestep = None): 
        if not self.dying and self.connected_dendrite is not None: 
            self.break_connection_program.reset()
            if type(self.connected_dendrite) == Axon:
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [self.connected_dendrite.parent_x_glob, 
                        self.connected_dendrite.parent_y_glob, 
                        self.connected_dendrite.parent_z_glob
                    ] + \
                    self.connected_dendrite.internal_states
            else:
                program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                    self.internal_states + \
                    [self.connected_dendrite.x, 
                        self.connected_dendrite.y, 
                        self.connected_dendrite.z
                    ] + \
                    [0 for x in range(len(self.internal_states))]
            outputs = self.break_connection_program.run(program_inputs)
            if randchoice(outputs[0]):
                if type(self.connected_dendrite) == Axon:
                    self.connected_dendrite.connected_dendrite = None
                    if not self.connected_dendrite.dying: 
                        self.connected_dendrite.neuron.grid.add_free_dendrite(self.connected_dendrite)
                        if timestep is not None: 
                            self.addqueue(
                                lambda: self.connected_dendrite.seek_dendrite_connection(), 
                                timestep + 1
                            )
                        else:
                            self.connected_dendrite.seek_dendrite_connection()
                else:
                    self.connected_dendrite.remove_subscriber(self)
                self.connected_dendrite = None
                if not self.dying: 
                    self.neuron.grid.add_free_dendrite(self)
                    if timestep is not None: 
                        self.addqueue(
                            lambda: self.seek_dendrite_connection(),
                            timestep + 1
                        )
                    else: 
                        self.seek_dendrite_connection()

    def run_recieve_reward(self, reward, timestep): 
        if not self.dying: 
            self.recieve_reward_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states + [reward]
            outputs = self.recieve_reward_program.run(program_inputs)
            if randchoice(outputs[0]):
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            if randchoice(outputs[1]):
                self.update_internal_state(outputs[2:2+self.internal_state_variable_count])
    
    def run_die(self, timestep):
        if not self.dying: 
            self.die_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states
            outputs = self.die_program.run(program_inputs)
            if randchoice(outputs[0]):
                self.dying = True
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[1:1+self.signal_arity], timestep),
                    timestep
                )
                self.addqueue(
                    lambda: self.run_signal_neuron(outputs[1:1+self.signal_arity], timestep),
                    timestep
                )
                self.addqueue(
                    lambda: self.die(timestep + 1), 
                    timestep + 1
                )

    def seek_dendrite_connection(self, timestep = None):
        dim_size = self.neuron_engine.get_size_in_neuron_positions_one_dim()
        max_x_dist = dim_size - self.parent_x_glob
        max_y_dist = dim_size - self.parent_y_glob
        max_z_dist = dim_size - self.parent_z_glob
        max_dist = max_x_dist + max_y_dist + max_z_dist # let's just do manhattan to start with because it is easier
        attempt = 0
        while attempt < self.seek_dendrite_tries:
            dist_target = int(math.floor((1-np.random.power(3))*max_dist))
            target_dendrite = self.neuron_engine.get_free_dendrite(self.neuron, dist_target)
            if target_dendrite is None: 
                break
            elif target_dendrite.run_accept_connection(self, timestep):
                if self.run_accept_connection(target_dendrite, timestep):
                    if not type(target_dendrite) == OutputNeuron and not type(target_dendrite) == InputNeuron:
                        target_dendrite.connected_dendrite = self
                        self.connected_dendrite = target_dendrite
                        target_dendrite.neuron.grid.remove_free_dendrite(target_dendrite)
                        self.neuron.grid.remove_free_dendrite(self)
                    else:
                        self.connected_dendrite = target_dendrite
                        target_dendrite.add_subscriber(self)
                    return True
            attempt += 1
        self.neuron.grid.add_free_dendrite(self)


    def run_action_controller(self, timestep):
        self.action_controller_program.reset()
        self.set_global_pos(self.action_controller_program, (0, 1, 2))
        self.set_internal_state_inputs(self.action_controller_program)
        outputs = self.action_controller_program.run_presetinputs()
        for num in range(self.program_order_length):
            output = outputs[num]
            if output >= 1.0:
                action = self.program_order_lambda_factory(timestep + 1, num) 
                # For some reason action is sometimes a lambda func inside a tuple. I have no idea why.
                if type(action) == tuple:
                    action = action[0]
                self.addqueue(
                    action,
                    timestep + 1
                )


    def die(self, timestep):
        self.dying = True
        self.neuron.grid.remove_free_dendrite(self)
        if self.connected_dendrite is not None:
            self.connected_dendrite.connected_dendrite = None
            if type(self.connected_dendrite) != OutputNeuron and type(self.connected_dendrite) != InputNeuron:
                self.connected_dendrite.neuron.grid.add_free_dendrite(self.connected_dendrite)
            self.connected_dendrite = None
        self.neuron.remove_dendrite(self)
