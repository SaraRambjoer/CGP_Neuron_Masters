# Setup problems
from CGP.genotype import Genome
import stupid_problem_test
problem = stupid_problem_test.StupidProblem()
# Setup logging

# Setup CGP genome
# - define a counter
from HelperClasses import Counter
counter = Counter()
neuron_internal_states = 3
dendrite_internal_states = 3
signal_dimensionality = 3
dimensions = 3
hox_variant_count = 1

# - define the function arities
# also define canonical order of functions - arbitrary, for compatibilitiy with 
# neuron code
# RFE move out this order to some single source of knowledge
neuron_function_order = [
    'axon_birth_program',
    'signal_axon_program',
    'recieve_axon_signal_program',
    'recieve_reward_program',
    'move_program',
    'die_program',
    'neuron_birth_program',
    'action_controller_program',
    'hox_variant_selection_program'
    'internal_state_variable_count', # not function but parameter comes here in the order
]
neuron_function_arities = [  # by order above
    [dimensions+neuron_internal_states+1, 3+signal_dimensionality],
    [signal_dimensionality+dimensions+neuron_internal_states, 2 + signal_dimensionality + neuron_internal_states],
    [signal_dimensionality + dimensions + neuron_internal_states, 3+neuron_internal_states+signal_dimensionality],
    [1 + dimensions + neuron_internal_states, 2 + neuron_internal_states],
    [neuron_internal_states + dimensions, 7+neuron_internal_states],
    [dimensions + neuron_internal_states, 2+neuron_internal_states],
    [dimensions + neuron_internal_states, 2+neuron_internal_states*2],
    [neuron_internal_states+dimensions, 9],
    [neuron_internal_states + dimensions, hox_variant_count]
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
    [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 2+dendrite_internal_states],
    [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 1],
    [dimensions + dendrite_internal_states + 1, 2 + dendrite_internal_states],
    [dimensions + dendrite_internal_states, 1+signal_dimensionality],
    [dendrite_internal_states + dimensions, 9]
]

# TODO add support for homeobox variant selection, currently just uses one
# Knowledge duplication thooo
def genome_to_init_data(genome):
    neuron_init_data = {
        'axon_birth_program' : genome.hex_selector_genome.program,
        'signal_axon_program' : genome.function_chromosomes[0].program,
        'recieve_axon_signal_program': genome.function_chromosomes[1].program,
        'recieve_reward_program': genome.function_chromosomes[2].program,
        'move_program': genome.function_chromosomes[3].program,
        'die_program': genome.function_chromosomes[4].program,
        'neuron_birth_program': genome.function_chromosomes[5].program,
        'action_controller_program': genome.function_chromosomes[6].program,
        'hox_variant_selection_program': genome.function_chromosomes[7].program,
        'internal_state_variable_count': neuron_internal_states
    }
    axon_init_data = {
        'recieve_signal_neuron_program' : genome.function_chromosomes[8].program,
        'recieve_signal_dendrite_program' : genome.function_chromosomes[9].program,
        'signal_dendrite_program' : genome.function_chromosomes[10].program,
        'signal_neuron_program' : genome.function_chromosomes[11].program,
        'accept_connection_program' : genome.function_chromosomes[12].program,
        'break_connection_program' : genome.function_chromosomes[13].program,
        'recieve_reward_program' : genome.function_chromosomes[14].program,
        'die_program' : genome.function_chromosomes[15].program,
        'action_controller_program' : genome.function_chromosomes[16].program,
    }
    return neuron_init_data, axon_init_data

# initialize the genome(s)
all_function_arities = neuron_function_arities + dendrite_function_arities
genome_count = 6
genomes = []
for num in range(genome_count):
    genomes.append(Genome(hox_variant_count, 4, all_function_arities, counter))


from neuron_engine.engine import NeuronEngine
# learning loop
import random
learning_iterations = 1000
genome_results = []
for genome in genomes:
    neuron_init, axon_init = genome_to_init_data(genome)
    engine = NeuronEngine(
        input_arity = problem.input_arity,
        output_arity = problem.output_arity,
        grid_count = 6,
        grid_size = 10,
        actions_max = 2000,
        neuron_initialization_data = neuron_init,
        axon_initialization_data = axon_init,
        signal_arity = signal_dimensionality,
        hox_variant_count = hox_variant_count,
        counter = counter,
        instances_per_iteration = 50
    )
    result = engine.run()
    genome_results.append(result)
genome = zip(genome, genome_results)
print(genome_results)

for num in range(learning_iterations):    
    new_genomes = []
    egligable_bachelors = [x[0] for x in genomes]
    while len(egligable_bachelors > 0):
        choice1 = random.choice(egligable_bachelors)
        egligable_bachelors.remove(choice1)
        choice2 = random.choice(egligable_bachelors)
        egligable_bachelors.remove(choice2)
        new_genomes += choice1.crossover(choice2)
    genome_results = []
    for genome in new_genomes:
        neuron_init, axon_init = genome_to_init_data(genome)
        engine = NeuronEngine(
            input_arity = problem.input_arity,
            output_arity = problem.output_arity,
            grid_count = 6,
            grid_size = 10,
            actions_max = 2000,
            neuron_initialization_data = neuron_init,
            axon_initialization_data = axon_init,
            signal_arity = signal_dimensionality,
            hox_variant_count = hox_variant_count,
            counter = counter,
            instances_per_iteration = 50
        )
        result = engine.run()
        genome_results.append(result)
    genomes = zip(new_genomes, genome_results)
    genomes.sort(key=lambda x: x[1])
    genomes = genomes[:genome_count]

    
