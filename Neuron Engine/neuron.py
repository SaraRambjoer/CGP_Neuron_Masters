
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
                counter) -> None:
        
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
            lambda t: self.run_hox_selection(t)
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


    # TODO add changing internal state variables on most of these mofos
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
                self.neuron_engine.add_action_to_queue(
                    lambda _: self.dendrites[-1].run_action_controller(), 
                    timestep + 1, 
                    self.id
                )
                
            if outputs[1] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_axon(outputs[2:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[2:], timestep + 1), 
                    timestep + 1, 
                    self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda _: self.axons[-1].run_action_controller(), 
                    timestep + 1,
                    self.id
                )
            
            if outputs[1] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_axon(outputs[2:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[2:], timestep + 1), 
                    timestep + 1, 
                    self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda: dendrite.run_recieve_signal_neuron(
                        outputs[1+self.internal_state_variable_count:], 
                        timestep + 1
                    ), 
                    timestep + 1, 
                    self.id
                )

        if outputs[1] >= 1.0:
            self.neuron_engine.add_action_to_queue(
                lambda: self.run_signal_axon(outputs[2:], timestep + 1), 
                timestep + 1, 
                self.id
            )
            self.neuron_engine.add_action_to_queue(
                lambda: self.run_signal_dendrite(outputs[2:], timestep + 1), 
                timestep + 1, 
                self.id
            )
    

    def run_signal_axon(self, signals, timestep):
        self.signal_axon_program.reset()
        self.set_signal_inputs(self.signal_axon_program, signals)
        self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals+3)))
        self.set_internal_state_inputs(self.signal_axon_program)
        outputs = self.signal_axon_program.run_presetinputs()

        for num in range(self.internal_state_variable_count):
            self.internal_states[num] += outputs[1 + num]

        if outputs[0] >= 1.0:
            for axon in self.axons:
                self.neuron_engine.add_action_to_queue(
                    lambda: axon.run_recieve_signal_neuron(
                        outputs[1 + self.internal_state_variable_count:], 
                        timestep + 1
                    ), 
                    timestep + 1, 
                    self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[2] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_axon(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )


            if outputs[0] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1, 
                    self.id
                )
    
    def run_recieve_signal_axon(self, signals, timestep):
        if not self.dying:
            self.recieve_axon_signal_program.reset()
            self.set_signal_inputs(self.recieve_axon_signal_program, signals)
            self.set_global_pos(self.signal_axon_program, range(len(signals), len(signals+3)))
            self.set_internal_state_inputs(self.recieve_axon_signal_program)
            outputs = self.recieve_axon_signal_program.run_presetinputs()

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[3 + num]

            if outputs[1] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[2] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_axon(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )


            if outputs[0] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1, 
                    self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1, 
                    self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_axon(outputs[7:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[7:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )
            
            self.x += x_translation_pos - x_translation_neg
            self.y += -y_translation_neg + y_translation_pos
            self.z += -z_translation_neg + z_translation_pos
            self.neuron_engine.update_neuron_position(self, (self.x, self.y, self.z))
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
                    self.neuron_engine.add_action_to_queue(
                        lambda: self.run_signal_axon(outputs[2:], timestep), 
                        timestep, 
                        self.id
                    )
                    self.neuron_engine.add_action_to_queue(
                        lambda: self.run_signal_dendrite(outputs[2:], timestep), 
                        timestep, 
                        self.id
                    )
    
    def run_neuron_birth(self, timestep):
        if not self.dying:
            self.neuron_birth_program.reset()
            self.set_global_pos(self.neuron_birth_program, (0, 1, 2))
            self.set_internal_state_inputs(self.neuron_birth_program)
            outputs = self.neuron_birth_program.run_presetinputs()
            if outputs[0] >= 1.0:
                new_neuron = self.neuron_engine.add_neuron((self.x, self.y, self.z), self.outputs[1:])
                self.neuron_engine.add_action_to_queue(
                    lambda: new_neuron.birth(timestep + 1), 
                    timestep + 1, 
                    new_neuron.id
                )
    

    def run_action_controller(self, timestep):
        if not self.dying:
            self.action_controller_program.reset()
            self.set_internal_state_inputs(self.action_controller_program)
            self.set_global_pos(self.action_controller_program, (0, 1, 2))
            outputs = self.action_controller_program.run_presetinputs()
            for num in range(len(outputs)):
                output = outputs[num]
                if output >= 1.0:
                    self.neuron_engine.add_action_to_queue(
                        lambda: self.program_order[num](timestep + 1),
                        timestep + 1,
                        self.id
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

# TODO consider merging axons and dendrites into a single object, would drastically reduce program size...
# Do this if search seems to not find anything
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

        # TODO fix 
        self.program_order = [
            self.recieve_signal_neuron_program,
            self.recieve_signal_dendrite_program,
            self.signal_dendrite_program,
            self.signal_neuron_program,
            self.accept_connection_program,
            self.break_connection_program,
            self.recieve_reward_program,
            self.die_program
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

            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[3 + num]

            if outputs[1] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[2] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_neuron(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[0] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1, 
                    self.id
                )

        elif self.connected_dendrite is None:
            self.seek_dendrite_connection()
    
    def run_recieve_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None:
            outputs = self.recieve_signal_setup(self.recieve_signal_axon_program, signals)
            
            for num in range(self.internal_state_variable_count):
                self.internal_states[num] += outputs[3 + num]

            if outputs[1] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_dendrite(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[2] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_signal_neuron(outputs[3:], timestep + 1), 
                    timestep + 1, 
                    self.id
                )

            if outputs[0] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1, 
                    self.id
                )
        
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
                self.neuron_engine.add_action_to_queue(
                    lambda: self.neuron.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1, 
                    self.id
                )
            if outputs[1+self.signal_arity] >= 1.0:
                for num in range(self.internal_state_variable_count):
                    self.internal_states[num] += outputs[1+self.signal_arity+1+num]
    
    def run_signal_dendrite(self, signals, timestep):
        if not self.dying and self.connected_dendrite is not None: 
            outputs = self.send_signal_setup(self.signal_dendrite_program, signals)
            if outputs[0] >= 1.0:
                self.neuron_engine.add_action_to_queue(
                    lambda: self.connected_dendrite.run_recieve_signal_axon(
                        outputs[1:1+self.signal_arity],
                        timestep + 1),
                    timestep + 1,
                    self.id
                )
            if outputs[1+self.signal_arity] >= 1.0:
                for num in range(self.internal_state_variable_count):
                    self.internal_states[num] += outputs[1+self.signal_arity+1+num]
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
                for num in range(self.internal_state_variable_count):
                    self.internal_states[num] += outputs[2 + num]
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
                    if timestep is not None: 
                        self.neuron_engine.add_action_to_queue(
                            lambda: self.connected_dendrite.seek_dendrite_connection(), 
                            timestep + 1, 
                            self.id
                        )
                    else:
                        self.connected_dendrite.seek_dendrite_connection()
                self.connected_dendrite = None
                if not self.dying: 
                    if timestep is not None: 
                        self.neuron_engine.add_action_to_queue(
                            lambda: self.seek_dendrite_connection(),
                            timestep + 1,
                            self.id
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
                self.neuron_engine.add_action_to_queue(
                    lambda: self.run_action_controller(timestep + 1), 
                    timestep + 1,
                    self.id
                )
            for num in range(self.internal_state_variable_count):
                self.internal_states[num] = outputs[2 + num]
    
    def run_die(self, timestep):
        if not self.dying: 
            self.die_program.reset()
            program_inputs = [self.parent_x_glob, self.parent_y_glob, self.parent_z_glob] + \
                self.internal_states
            outputs = self.die_program.run(program_inputs)
            if outputs[0] >= 1.0:
                self.dying = True
                self.neuron_engine.add_action_to_queue(lambda: self.die(timestep + 1), timestep + 1, self.id)

    def seek_dendrite_connection(self):
        pass # TODO, use power law somehow using grids

    def set_internal_state_inputs(self, program):
        num = 0
        for input_node in program.inputs[-self.internal_state_variable_count:]:
            input_node.set_output(self.internal_states[num])
            num += 1
    
    def execute_action_controller_output(self, outputs, timestep):
        for num in range(len(outputs)):
            output = outputs[num]
            if output >= 1.0:
                self.neuron_engine.add_action_to_queue(self.program_order[num], timestep + 1, self.id)

    def die(self, timestep):
        self.dying = True
        
        # send death signal add death to queue at timestep
        pass
