 class Func:
     def __init__(self) -> None:
         pass


# neuron axon birth
# [dimensions+neuron_internal_states+1, 4+signal_dimensionality+neuron_internal_states],  # axon birth
# until internal states 0 is 1

 
 
 neuron_function_order = [
        'axon_birth_program',
        'signal_axon_program',
        'recieve_axon_signal_program',
        'recieve_reward_program',
        'move_program',
        'die_program',
        'neuron_birth_program',
        'action_controller_program',
        'hox_variant_selection_program',
        'internal_state_variable_count' # not function but parameter comes here in the order
    ]
    neuron_function_arities = [  # by order above
        [dimensions+neuron_internal_states+1, 4+signal_dimensionality+neuron_internal_states],  # axon birth
        [signal_dimensionality+dimensions+neuron_internal_states, 2 + signal_dimensionality + neuron_internal_states],  # signal axon
        [signal_dimensionality + dimensions + neuron_internal_states, 4+neuron_internal_states+signal_dimensionality],  # recieve signal axon
        [1 + dimensions + neuron_internal_states, 2 + neuron_internal_states],  # reciee reward
        [neuron_internal_states + dimensions, 7+neuron_internal_states],  # move
        [dimensions + neuron_internal_states, 2+neuron_internal_states],  # die
        [dimensions + neuron_internal_states, 2+neuron_internal_states*2],  # neuron birth
        [neuron_internal_states+dimensions, 9],  # action controller
        [neuron_internal_states + dimensions, hox_variant_count]  # hox selection
    ]

    dendrite_function_order = [
        'recieve_signal_neuron_program',
        'recieve_signal_dendrite_program',
        'signal_dendrite_program',
        'signal_neuron_program',
        'accept_connection_program',
        'break_connection_program',
        'recieve_reward_program',
        'die_program',
        'action_controller_program'
    ]
    dendrite_function_arities = [
        [dendrite_internal_states + signal_dimensionality + dimensions, 4+signal_dimensionality+dendrite_internal_states],
        [dendrite_internal_states + signal_dimensionality + dimensions, 4+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + signal_dimensionality, 4+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + signal_dimensionality, 4+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 2+dendrite_internal_states], # Accept connection
        [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 1], # Break connection
        [dimensions + dendrite_internal_states + 1, 2 + dendrite_internal_states], # recieve reward
        [dimensions + dendrite_internal_states, 1+signal_dimensionality], # die
        [dendrite_internal_states + dimensions, 3]
    ]

    # TODO add support for homeobox variant selection, currently just uses one
    # Knowledge duplication thooo
    def genome_to_init_data(genome):
        neuron_init_data = {
            'axon_birth_program' : genome.function_chromosomes[0].hex_variants[0].program,
            'signal_axon_program' : genome.function_chromosomes[1].hex_variants[0].program,
            'recieve_axon_signal_program': genome.function_chromosomes[2].hex_variants[0].program,
            'recieve_reward_program': genome.function_chromosomes[3].hex_variants[0].program,
            'move_program': genome.function_chromosomes[4].hex_variants[0].program,
            'die_program': genome.function_chromosomes[5].hex_variants[0].program,
            'neuron_birth_program': genome.function_chromosomes[6].hex_variants[0].program,
            'action_controller_program': genome.function_chromosomes[7].hex_variants[0].program,
            'hox_variant_selection_program': genome.hex_selector_genome.program,
            'internal_state_variable_count': neuron_internal_states
        }
        axon_init_data = {
            'recieve_signal_neuron_program' : genome.function_chromosomes[8].hex_variants[0].program,
            'recieve_signal_dendrite_program' : genome.function_chromosomes[9].hex_variants[0].program,
            'signal_dendrite_program' : genome.function_chromosomes[10].hex_variants[0].program,
            'signal_neuron_program' : genome.function_chromosomes[11].hex_variants[0].program,
            'accept_connection_program' : genome.function_chromosomes[12].hex_variants[0].program,
            'break_connection_program' : genome.function_chromosomes[13].hex_variants[0].program,
            'recieve_reward_program' : genome.function_chromosomes[14].hex_variants[0].program,
            'die_program' : genome.function_chromosomes[15].hex_variants[0].program,
            'action_controller_program' : genome.function_chromosomes[16].hex_variants[0].program,
            'internal_state_variable_count': dendrite_internal_states
        }
        return neuron_init_data, axon_init_data
