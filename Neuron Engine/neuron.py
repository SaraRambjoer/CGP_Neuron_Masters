
import math
import numpy as np 

def addqueue(neuron_engine, lambdafunc, timestep, id):
    neuron_engine.add_action_to_queue(
        lambdafunc,
        timestep,
        id
    )
# TODO change action controller s.t. it can generate and send signals?
# RFE some code duplication issues
# also there is going ot be some knowledge duplication in terms of length of inputs and outputs for each program
class Neuron():
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
        self.internal_state_variable_count = self.neuron_initialization_data['internal_state_variable_count']
        self.hox_variant_selection_program = self.neuron_initialization_data['hox_variant_selection_program']

        self.program_order = [
            lambda t: self.run_dendrite_birth(t),
            lambda t: self.run_axon_birth(t),
            lambda t: self.run_move(t),
            lambda t: self.run_die(t),
            lambda t: self.run_neuron_birth(t),
            lambda t: self.run_hox_selection(t),
            lambda t: self.run_action_controller(t)
        ]

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

    def addqueue(self, lambdafunc, timestep):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id)

    def update_internal_state(self, deltas):
        for num in range(self.internal_state_variable_count):
            self.internal_states[num] += deltas

    # TODO add changing internal state variables on most of these mofos (WIP)
    def run_hox_selection(self):
        if not self.dying:
            self.hox_variant_selection_program.reset()
            self.set_internal_state_inputs(self.hox_variant_selection_program)
            self.set_global_pos(self.hox_variant_selection_program(0, 1, 2))
            outputs = self.hox_variant_selection_program.run_presetinputs()
            chosen_variant = outputs.index(max(outputs))
            self.selected_hox_variant = chosen_variant

    def set_global_pos(self, program, indexes):
        program.input_nodes[indexes[0]] = self.x_glob
        program.input_nodes[indexes[1]] = self.y_glob
        program.input_nodes[indexes[2]] = self.z_glob

    def run_dendrite_birth(self, timestep):
        if not self.dying:
            self.axon_birth_program.reset()
            self.set_internal_state_inputs(self.axon_birth_program)
            self.set_global_pos(self.axon_birth_program, (0, 1, 2))
            self.axon_birth_program.input_nodes[3].set_output(len(self.dendrites))
            outputs = self.axon_birth_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if outputs[0] >= 1.0:
                self.dendrites.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self))
                self.addqueue(
                    lambda: self.dendrites[-1].run_action_controller(),
                    timestep + 1
                )
                
            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
            
            if outputs[2+self.signal_arity] >= 1.0:
                self.update_internal_state(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count])
            
            if outputs[-1] >= 1.0:
                self.addqueue(
                    self.run_action_controller(timestep+1),
                    timestep + 1
                )

     
    def run_axon_birth(self, timestep):
        if not self.dying:
            self.axon_birth_program.reset()
            self.set_internal_state_inputs(self.axon_birth_program)
            self.set_global_pos(self.axon_birth_program, (0, 1, 2))
            self.axon_birth_program.input_nodes[3].set_output(len(self.axons))
            outputs = self.axon_birth_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if outputs[0] >= 1.0:
                self.axons.append(Axon(self.axon_initialization_data, self.neuron_engine, self.signal_arity, self.counter, self))
                self.addqueue(
                    lambda _: self.axons[-1].run_action_controller(timestep+1), 
                    timestep + 1
                )
            
            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[2:2+self.signal_arity], timestep + 1), 
                    timestep + 1
                )
            
            if outputs[2+self.signal_arity] >= 1.0:
                self.update_internal_state(outputs[3+self.signal_arity:3+self.signal_arity+self.internal_state_variable_count])
            
            if outputs[-1] >= 1.0:
                self.addqueue(
                    self.run_action_controller(timestep + 1),
                    timestep + 1
                )
    
    def set_signal_inputs(self, program, signals):
        for num in range(len(signals)):
            program.input_nodes[num].set_output(signals[num])

    def run_signal_dendrite(self, signals, timestep):
        self.signal_axon_program.reset()
        self.set_signal_inputs(self.signal_axon_program, signals)
        self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals)+3))
        self.set_internal_state_inputs(self.signal_axon_program)
        outputs = self.signal_axon_program.run_presetinputs()

        for num in range(self.internal_state_variable_count):
            self.internal_states[num] += outputs[1 + num]

        if outputs[0] >= 1.0:
            for dendrite in self.dendrites:
                self.addqueue(
                    lambda: dendrite.run_recieve_signal_neuron(
                        outputs[1:1+self.signal_arity], 
                        timestep + 1
                    ), 
                    timestep + 1
                )
        
        if outputs[1+self.signal_arity] >= 1.0:
            self.update_internal_state(outputs[1+self.signal_arity:1+self.signal_arity+self.internal_state_variable_count])
        
        if outputs[-1] >= 1.0:
            self.addqueue(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1
            )
    

    def run_signal_axon(self, signals, timestep):
        self.signal_axon_program.reset()
        self.set_signal_inputs(self.signal_axon_program, signals)
        self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals+3)))
        self.set_internal_state_inputs(self.signal_axon_program)
        outputs = self.signal_axon_program.run_presetinputs()

        if outputs[0] >= 1.0:
            for axon in self.axons:
                self.addqueue(
                    lambda: axon.run_recieve_signal_neuron(
                        outputs[1 + self.internal_state_variable_count:], 
                        timestep + 1
                    ), 
                    timestep + 1
                )
        
        if outputs[1+self.signal_arity] >= 1.0:
            self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
        
        if outputs[-1]:
            self.addqueue(
                lambda: self.run_action_controller(timestep + 1),
                timestep + 1
            )

    
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying: 
            self.recieve_axon_signal_program.reset()
            self.set_signal_inputs(self.recieve_axon_signal_program, signals)
            self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals+3)))
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[3 + num]

            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:self.signal_arity+3], timestep + 1), 
                    timestep + 1
                )

            if outputs[2] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[3:self.signal_arity+3], timestep + 1), 
                    timestep + 1
                )


            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            
            if outputs[3+self.signal_arity] >= 1.0:
                self.update_internal_state(outputs[3+self.signal_arity:3+self.internal_state_variable_count+self.signal_arity])
    
    def run_recieve_signal_axon(self, signals, timestep):
        if not self.dying:
            self.recieve_axon_signal_program.reset()
            self.set_signal_inputs(self.recieve_axon_signal_program, signals)
            self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals+3)))
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            self.update_internal_state(outputs[3:3+self.internal_state_variable_count])

            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:], timestep + 1), 
                    timestep + 1
                )

            if outputs[2] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[3:], timestep + 1), 
                    timestep + 1
                )


            if outputs[0] >= 1.0:
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
            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[1 + num]

            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )

    def run_move(self, timestep):
        if not self.dying:
            self.move_program.reset()
            self.set_internal_state_inputs(self.move_program)
            self.set_global_pos(self.move_program, (0, 1, 2))
            outputs = self.move_prgram.run_presetinputs()
            x_translation_pos = 1.0 if outputs[0] >= 1.0 else 0.0
            y_translation_pos = 1.0 if outputs[1] >= 1.0 else 0.0
            z_translation_pos = 1.0 if outputs[2] >= 1.0 else 0.0
            x_translation_neg = 1.0 if outputs[3] <= -1.0 else 0.0
            y_translation_neg = 1.0 if outputs[4] <= -1.0 else 0.0
            z_translation_neg = 1.0 if outputs[5] <= -1.0 else 0.0

            if outputs[6] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_axon(outputs[7:], timestep + 1), 
                    timestep + 1
                )
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[7:], timestep + 1), 
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
            if outputs[0] >= 1.0:
                self.dying = True
                self.neuron_engine.remove_neuron(self.id, timestep + 1)
                for axon in self.axons:
                    axon.die(timestep + 1)
                for dendrite in self.dendrites:
                    dendrite.die(timestep + 1)
            
                # send death signal optionally
                if outputs[1] >= 1.0:
                    # notice doing it at current timestep, breaking normal sequence, because dendrite.die is called at next timestep
                    self.addqueue(
                        lambda: self.run_signal_axon(outputs[2:], timestep), 
                        timestep
                    )
                    self.addqueue(
                        lambda: self.run_signal_dendrite(outputs[2:], timestep), 
                        timestep
                    )
    
    def run_neuron_birth(self, timestep):
        if not self.dying:
            self.neuron_birth_program.reset()
            self.set_global_pos(self.neuron_birth_program, (0, 1, 2))
            self.set_internal_state_inputs(self.neuron_birth_program)
            outputs = self.neuron_birth_program.run_presetinputs()
            if outputs[0] >= 1.0:
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
                new_neuron.internal_states = self.outputs[1:1+new_neuron.internal_state_variable_count]
                new_neuron = self.neuron_engine.add_neuron(new_neuron)
                if outputs[1+new_neuron.internal_state_variable_count] >= 1.0:
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
            self.set_internal_state_inputs(self.action_controller_program)
            self.set_global_pos(self.action_controller_program, (0, 1, 2))
            outputs = self.action_controller_program.run_presetinputs()
            for num in range(len(self.program_order)):
                output = outputs[num]
                if output >= 1.0:
                    self.addqueue(
                        lambda: self.program_order[num](timestep + 1),
                        timestep + 1
                    )


    def set_internal_state_inputs(self, program):
        num = 0
        for input_node in program.inputs[-self.internal_state_variable_count:]:
            input_node.set_output(self.internal_states[num])
            num += 1
    
    def execute_action_controller_output(self, outputs, timestep):
        for num in range(len(outputs)):
            output = outputs[num]
            if output >= 1.0:
                self.add_to_queue(num, timestep + 1)
    
    def remove_dendrite(self, dendrite):
        if dendrite in self.axons:
            self.axons.remove(dendrite)
        if dendrite in self.dendrites:
            self.dendrites.remove(dendrite)
        self.neuron_engine.remove_dendrite(dendrite)

class Axon():
    def __init__(self,
                 axon_initialization_data,
                 neuron_engine,
                 signal_arity,
                 counter,
                 neuron) -> None:

        self.signal_arity = signal_arity
        self.counter = counter
        self.id = self.counter.counterval()

        self.recieve_signal_neuron_program = axon_initialization_data['recieve_signal_neuron_program']
        self.recieve_signal_dendrite_program = axon_initialization_data['recieve_signal_dendrite_program']
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
    
    def addqueue(self, lambdafunc, timestep):
        addqueue(self.neuron_engine, lambdafunc, timestep, self.id)

    def update_internal_state(self, deltas):
        for num in range(self.internal_state_variable_count):
            self.internal_states[num] += deltas

    def set_global_pos(self, program, indexes):
        program.input_nodes[indexes[0]] = self.parent_x_glob
        program.input_nodes[indexes[1]] = self.parent_y_glob
        program.input_nodes[indexes[2]] = self.parent_z_glob

    def update_pos(self, x, y, z):
        self.parent_x_glob = x
        self.parent_y_glob = y
        self.parent_z_glob = z

    # TODO when run break connection? Think maybe run action controller a lot

    def recieve_signal_setup(self, program, signals):
        program.reset()
        self.set_internal_state_inputs(program)
        for num in range(0, len(signals)):
            program.inputs.set_output(signals[num])
        self.set_global_pos(program, range(len(signals), len(signals)+3))
        outputs = program.run_presetinputs()
        return outputs

    def run_recieve_signal_neuron(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None:
            outputs = self.recieve_signal_setup(self.recieve_signal_neuron_program, signals)
            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if outputs[2] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_neuron(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            if outputs[self.signal_arity+3] >= 1.0:
                self.update_internal_state(outputs[self.signal_arity+4:self.signal_arity+self.internal_state_variable_count+4])

        elif self.connected_dendrite is None:
            self.seek_dendrite_connection()
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None:
            outputs = self.recieve_signal_setup(self.recieve_signal_axon_program, signals)
        
            if outputs[1] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_dendrite(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if outputs[2] >= 1.0:
                self.addqueue(
                    lambda: self.run_signal_neuron(outputs[3:3+self.signal_arity], timestep + 1), 
                    timestep + 1
                )

            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            
            if outputs[self.signal_arity+3] >= 1.0:
                self.update_internal_state(outputs[self.signal_arity+4:self.signal_arity+self.internal_state_variable_count+4])
        
        elif self.connected_dendrite is None:
            self.seek_dendrite_connection()
    
    def send_signal_setup(self, program, signals):
        program.reset()
        self.set_global_pos(program, (0, 1, 2))
        self.set_internal_state_inputs(program)
        for num in range(self.signal_arity):
            program.inputs[num+3].set_output(signals[num])
        return program.run_presetinputs()

    def run_signal_neuron(self, signals, timestep):
        if not self.dying: 
            outputs = self.send_signal_setup(self.signal_neuron_program, signals)
            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.neuron.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1
                )
            if outputs[1+self.signal_arity] >= 1.0:
                self.update_internal_state(outputs[2+self.signal_arity:2+self.signal_arity+self.internal_state_variable_count])
            if outputs[-1] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep+1),
                    timestep + 1
                )

    def run_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None: 
            outputs = self.send_signal_setup(self.signal_dendrite_program, signals)
            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.connected_dendrite.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1
                )
            if outputs[1+self.signal_arity] >= 1.0:
                self.update_internal_state(outputs[2+self.signal_arity+2+self.signal_arity+self.internal_state_variable_count])
            if outputs[-1] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1),
                    timestep + 1
                )
            
        if self.connected_dendrite is None:
            self.seek_dendrite_connection(timestep)
        
    def run_accept_connection(self, dendrite, timestep = None):
        if not self.dying and self.connected_dendrite is None:
            self.accept_connection_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states + \
                [dendrite.parent_x_glob, dendrite.parent_y_glob, dendrite.parent_z_glob] + \
                dendrite.internal_states
            outputs = self.accept_connection_program.run(program_inputs)
            if outputs[0] >= 1.0:
                return True
            if outputs[1] >= 1.0:
                self.update_internal_state(outputs[2:2+self.internal_state_variable_count])
            return False
        return False # shouldn't happen maybe check for this TODO indicates fault in seek_dendrite_connection code
    
    def run_break_connection(self, timestep = None): 
        if not self.dying and self.connected_dendrite is not None: 
            self.break_connection_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states + \
                [self.conneced_dendrite.parent_x_glob, 
                    self.conneced_dendrite.parent_y_glob, 
                    self.conneced_dendrite.parent_z_glob
                ] + \
                self.conneced_dendrite.internal_states
            outputs = self.break_connection_program.run(program_inputs)
            if outputs[0] >= 1.0:
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
            if outputs[0] >= 1.0:
                self.addqueue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1
                )
            self.update_internal_state(outputs[2:2+self.internal_state_variable_count])
    
    def run_die(self, timestep):
        if not self.dying: 
            self.die_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states
            outputs = self.die_program.run(program_inputs)
            if outputs[0] >= 1.0:
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
                    target_dendrite.connected_dendrite = self
                    self.connected_dendrite = target_dendrite
                    target_dendrite.neuron.grid.remove_free_dendrite(target_dendrite)
                    return True
        self.neuron.grid.add_free_dendrite(self)


    def set_internal_state_inputs(self, program):
        num = 0
        for input_node in program.inputs[-self.internal_state_variable_count:]:
            input_node.set_output(self.internal_states[num])
            num += 1
    
    def run_action_controller(self, timestep):
        self.action_controller_program.reset()
        self.set_internal_state_inputs(self.action_controller_program)
        self.set_global_pos(self.action_controller_program, (0, 1, 2))
        outputs = self.action_controller_program.run_presetinputs()
        for num in range(len(self.program_order)):
            output = outputs[num]
            if output >= 1.0:
                self.neuron_engine.addqueue(
                    lambda: self.program_order[num](timestep + 1),
                    timestep + 1
                )


    def die(self, timestep):
        self.dying = True
        self.neuron.grid.remove_free_dendrite(self)
        if self.connected_dendrite is not None:
            self.connected_dendrite.connected_dendrite = None
            self.connected_dendrite.neuron.grid.add_free_dendrite(self.connected_dendrite)
            self.connected_dendrite = None
        self.neuron.remove_dendrite(self)
