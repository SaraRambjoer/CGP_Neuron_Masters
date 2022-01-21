import time
import json
from engine import NeuronEngine
from genotype import Genome
import Logger
import stupid_problem_test
import random
from HelperClasses import Counter, randchoice, drawProgram, copydict
import os

def log_genome(genomes, runinfo):
    for genome in genomes:
        initial_data = {
            "genome id" : genome[0].id,
            "genome fitness" : genome[1],
            "run" : runinfo
        }
        genome[0].log(initial_data)

def process_config(config):
    config = dict(config)
    for key, val in config.items():
        if "," in val:
            config[key] = val.split(',')
        elif "." in val:
            config[key] = float(val)
        elif config[key] == "False":
            config[key] = False
        elif config[key] == "True":
            config[key] = True
        else: 
            nums = [str(x) for x in range(0, 10)]
            for num in nums:
                if num in val:
                    config[key] = int(val)
                    break
    return config


def run(config, print_output = False):
    # Setup problems
    problem = stupid_problem_test.StupidProblem()
    # Setup logging
    # ["CGPProgram image", "cgp_function_exec_prio1", "cgp_function_exec_prio2", "graphlog_instance", "graphlog_run", "setup_info"]
    logger = Logger.Logger(os.path.join(os.path.dirname(__file__), "logfiles") + "\\log", config['logger_ignore_messages'])
    # Setup CGP genome
    # - define a counter
    counter = Counter()
    neuron_internal_states = config['neuron_internal_state_count']
    dendrite_internal_states = config['axon_dendrite_internal_state_count']
    signal_dimensionality = config['signal_dimensionality']
    dimensions = 3  # other dimensions not supported - code in engine.py specific to 3d grid
    hox_variant_count = config['hox_variant_count']
    genome_counter = Counter()
    genome_count = config['genome_count']
    seed = config['seed']
    random.seed(seed)

    logger.log_json("setup_info", dict(config))

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
        'hox_variant_selection_program',
        'internal_state_variable_count' # not function but parameter comes here in the order
    ]
    neuron_function_arities = [  # by order above
        [dimensions+neuron_internal_states+1, 4+signal_dimensionality+neuron_internal_states],  # axon birth
        [signal_dimensionality+dimensions+neuron_internal_states, 2 + signal_dimensionality + neuron_internal_states],  # signal axon
        [signal_dimensionality + dimensions + neuron_internal_states, 2 + neuron_internal_states+signal_dimensionality],  # recieve signal axon
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
        [dendrite_internal_states + signal_dimensionality + dimensions, 2+signal_dimensionality+dendrite_internal_states],
        [dendrite_internal_states + signal_dimensionality + dimensions, 2+signal_dimensionality+dendrite_internal_states],
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
            'axon_birth_programs' : genome.function_chromosomes[0],
            'signal_axon_programs' : genome.function_chromosomes[1],
            'recieve_axon_signal_programs': genome.function_chromosomes[2],
            'recieve_reward_programs': genome.function_chromosomes[3],
            'move_programs': genome.function_chromosomes[4],
            'die_programs': genome.function_chromosomes[5],
            'neuron_birth_programs': genome.function_chromosomes[6],
            'action_controller_programs': genome.function_chromosomes[7],
            'hox_variant_selection_program': genome.hex_selector_genome.program,
            'internal_state_variable_count': neuron_internal_states
        }
        axon_init_data = {
            'recieve_signal_neuron_programs' : genome.function_chromosomes[8],
            'recieve_signal_dendrite_programs' : genome.function_chromosomes[9],
            'signal_dendrite_programs' : genome.function_chromosomes[10],
            'signal_neuron_programs' : genome.function_chromosomes[11],
            'accept_connection_programs' : genome.function_chromosomes[12],
            'break_connection_programs' : genome.function_chromosomes[13],
            'recieve_reward_programs' : genome.function_chromosomes[14],
            'die_programs' : genome.function_chromosomes[15],
            'action_controller_programs' : genome.function_chromosomes[16],
            'internal_state_variable_count': dendrite_internal_states
        }
        return neuron_init_data, axon_init_data

    import CGPEngine
    genome_successor_count = 4
    if not config['non_crossover_children']:
        genome_successor_count = 2
    # initialize the genome(s)
    all_function_arities = neuron_function_arities + dendrite_function_arities
    genomes = []
    for num in range(genome_count):
        genomes.append(Genome(
            homeobox_variants = hox_variant_count,
            successor_count = genome_successor_count,
            input_arities = all_function_arities,
            counter = counter,
            internal_state_variables = neuron_internal_states,
            names = neuron_function_order[:-1] + dendrite_function_order,
            logger = logger,
            genome_counter = genome_counter,
            config = config)) # TODO RN assumes equal amount of axon and neuron internal state variables


    from engine import NeuronEngine
    # learning loop

    to_return_fitness = []

    learning_iterations = config['iterations']
    genome_results = []
    neuron_init, axon_init = genome_to_init_data(genomes[0])
    for genome in genomes:
        engine = NeuronEngine(
            input_arity = problem.input_arity,
            output_arity = problem.output_arity,
            grid_count = 6,
            grid_size = 10,
            actions_max = 120,
            neuron_initialization_data = neuron_init,
            axon_initialization_data = axon_init,
            signal_arity = signal_dimensionality,
            hox_variant_count = hox_variant_count,
            instances_per_iteration = 50,
            logger = logger,
            genome_id = genome.id,
            config_file = copydict(config)
        )
        result, base_problems = engine.run(problem, "setup")
        genome_results.append((result, base_problems))
    genomes = list(zip(genomes, [x[0] for x in genome_results], [x[1] for x in genome_results]))
    to_return_fitness.append(x[0] for x in genome_results)
    log_genome(genomes, 0)
    for num in range(learning_iterations):   
        time_genes = 0
        time_eval = 0
        egligable_bachelors = [x[0] for x in genomes]
        child_data = [[] for _ in range(len(genomes))]
        while len([x for x in egligable_bachelors if x is not None]) > 0:
            time_genes_stamp = time.time()
            choice1 = randchoice([x for x in egligable_bachelors if x is not None])
            choice2 = randchoice([x for x in egligable_bachelors if x is not None])
            indexes = [egligable_bachelors.index(choice1), egligable_bachelors.index(choice2)]
            egligable_bachelors[egligable_bachelors.index(choice1)] = None  # Currently possible to do crossover with self, which does make some sense with subgraph extraction
            if choice2 in egligable_bachelors and choice2 != choice1:
                egligable_bachelors[egligable_bachelors.index(choice2)] = None
            new_genomes = choice1.crossover(choice2)
            skip_eval = [False for num in range(len(new_genomes))]
            for numero in range(len(new_genomes)):
                genome = new_genomes[numero]
                if genome.equals_no_id(choice1):
                    skip_eval[numero] = 1
                if genome.equals_no_id(choice2):
                    skip_eval[numero] = 2
            genome_results = []
            genome_results = []
            for numero in range(len(new_genomes)):
                genome = new_genomes[numero]
                time_genes += time.time() - time_genes_stamp
                time_eval_stamp = time.time()
                if not skip_eval[numero]:
                    neuron_initialization_data, axon_initialization_data = genome_to_init_data(genome)
                    engine = NeuronEngine(
                        input_arity = problem.input_arity,
                        output_arity = problem.output_arity,
                        grid_count = 6,
                        grid_size = 10,
                        actions_max = 120,
                        neuron_initialization_data = neuron_initialization_data,
                        axon_initialization_data = axon_initialization_data,
                        signal_arity = signal_dimensionality,
                        hox_variant_count = hox_variant_count,
                        instances_per_iteration = 50,
                        logger = logger,
                        genome_id = genome.id,
                        config_file = copydict(config)
                    )
                    genome_results.append(engine.run(problem, num))
                elif skip_eval[numero] == 1:
                    genome_results.append((genomes[indexes[0]][1], genomes[indexes[0]][2]))
                else:
                    genome_results.append((genomes[indexes[1]][1], genomes[indexes[1]][2]))
                time_eval += time.time() - time_eval_stamp

            #def multiprocess_code(engine_problem):
            #    return engine_problem[0].run(engine_problem[1])
          #
            #with Pool() as p:
            #    results = p.map(multiprocess_code, list(zip(new_genomes, [stupid_problem_test.StupidProblem() for _ in range(len(new_genomes))])))
            time_genes_stamp = time.time()
            base_problems = [x[1] for x in genome_results]
            genome_results = [x[0] for x in genome_results]
            # all children of a parent compete for the parents spots

            def _draw_program_data(genome):
                # RFE only one hex variant shown
                genome = genome
                functions = genome.function_chromosomes
                for func in functions:
                    chro = func
                    program = chro.hex_variants[0].program
                    output_nodes = [program.nodes[x] for x in program.output_indexes]
                    drawProgram(
                        program.get_active_nodes(),
                        output_nodes,
                        program.input_nodes
                    )

            for x in range(len(new_genomes)):
                child_data[indexes[0]].append((new_genomes[x], genome_results[x], base_problems[x]))
                child_data[indexes[1]].append((new_genomes[x], genome_results[x], base_problems[x]))        
            time_genes += time.time() - time_genes_stamp

        time_genes_stamp = time.time()
        change_better = [False for x in range(len(genomes))]
        change_neutral = [False for x in range(len(genomes))]
        for num3 in range(len(child_data)):
            score_view = [x[1] for x in child_data[num3]]
            score_min = min(score_view)
            min_index = score_view.index(score_min)
            if score_min <= genomes[num3][1]:
                if score_min < genomes[num3][1]:
                    change_better[num3] = True
                else:
                    change_neutral[num3] = True
                genomes[num3] = child_data[num3][min_index]

        
        for num3 in range(len(genomes)):
            genome = genomes[num3][0]
            x = (genome.config['mutation_chance_node']+genome.config['mutation_chance_link'])/2
            genome.config['mutation_chance_link'] = x
            genome.config['mutation_chance_node'] = x
            if change_better[num3]:
                pass
            elif change_neutral[num3]:
                genome.config['mutation_chance_node'] = min(genome.config['max_mutation_chance_node'], genome.config['mutation_chance_node']*config['neutral_mutation_chance_node_multiplier'])
                genome.config['mutation_chance_link'] = min(genome.config['max_mutation_chance_link'], genome.config['mutation_chance_link']*config['neutral_mutation_chance_link_multiplier'])

            else:
                if not(genome.hypermutation):
                    genome.config['mutation_chance_node'] *= config['fail_mutation_chance_node_multiplier']
                    genome.config['mutation_chance_link'] *= config['fail_mutation_chance_link_multiplier']
                    if genome.config['mutation_chance_node'] < 0.000001:
                        genome.hypermutation = True
                else:
                    genome.config['mutation_chance_node'] = config['hypermutation_mutation_chance']
                    genome.config['mutation_chance_link'] = config['hypermutation_mutation_chance']
            genome.update_config()

        time_genes += time.time() - time_genes_stamp
        #print(num, [f"{x[1]}, {x[2]}" for x in genomes])
        print(f"------------------- {num} ------------------")
        print(time_genes, time_eval)
        for genome in genomes: 
            print()
            print(genome[0].id)
            print(genome[1])
            print(genome[2])
            print(genome[0].config['mutation_chance_node'], genome[0].config['mutation_chance_link'])
        to_return_fitness.append([x[1] for x in genomes])
        log_genome(genomes, num)
        #_genomes = [x[0] for x in genomes]
        #for gen in _genomes:
        #  print(str(gen))
    return to_return_fitness

if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config["Default"]
    config = process_config(config)
    if config['mode'] == 'run':
        print("Running evolution")
        run(config, print_output=True)
    elif config['mode'][0] == 'load':
        # TODO not fully implemented
        print("Loading program")
        loadfile = config['mode'][1]
        loadprogram = config['mode'][2]

        # get specific cgp program

        with open(loadfile, 'r') as f:
            data = f.readlines()

        data = data[0]
        
        genomes = data.split('|')

        correct_genome = None
        for genome in genomes:
            gene_dat = json.loads(genome)
            if gene_dat['genome_id'].split("->")[1][1:] == loadprogram:
                correct_genome = gene_dat
                break
        
        if correct_genome is None:
            print(f"Genome {loadprogram} not found")
        else:
            print("Genome found")
            neuron_internal_states = config['neuron_internal_state_count']
            dendrite_internal_states = config['axon_dendrite_internal_state_count']
            signal_dimensionality = config['signal_dimensionality']
            dimensions = 3  # other dimensions not supported - code in engine.py specific to 3d grid
            hox_variant_count = config['hox_variant_count']
            genome_counter = Counter()
            genome_count = config['genome_count']
            seed = config['seed']
            random.seed(seed)
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
                [signal_dimensionality + dimensions + neuron_internal_states, 2 + neuron_internal_states+signal_dimensionality],  # recieve signal axon
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
                [dendrite_internal_states + signal_dimensionality + dimensions, 2+signal_dimensionality+dendrite_internal_states],
                [dendrite_internal_states + signal_dimensionality + dimensions, 2+signal_dimensionality+dendrite_internal_states],
                [dimensions + dendrite_internal_states + signal_dimensionality, 4+signal_dimensionality+dendrite_internal_states],
                [dimensions + dendrite_internal_states + signal_dimensionality, 4+signal_dimensionality+dendrite_internal_states],
                [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 2+dendrite_internal_states], # Accept connection
                [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states, 1], # Break connection
                [dimensions + dendrite_internal_states + 1, 2 + dendrite_internal_states], # recieve reward
                [dimensions + dendrite_internal_states, 1+signal_dimensionality], # die
                [dendrite_internal_states + dimensions, 3]
            ]
            logger = Logger.Logger(os.path.join(os.path.dirname(__file__), "logfiles") + "\\log", config['logger_ignore_messages'])
            genome_successor_count = 4
            if not config['non_crossover_children']:
                genome_successor_count = 2
            all_function_arities = neuron_function_arities + dendrite_function_arities

            genome = Genome(
                homeobox_variants = hox_variant_count,
                successor_count = genome_successor_count,
                input_arities = all_function_arities,
                counter = genome_counter,
                internal_state_variables = neuron_internal_states,
                names = neuron_function_order[:-1] + dendrite_function_order,
                logger = logger,
                genome_counter = genome_counter,
                config = config)

            genome.load(correct_genome)

            problem = stupid_problem_test.StupidProblem()

            def genome_to_init_data(genome):
                neuron_init_data = {
                    'axon_birth_programs' : genome.function_chromosomes[0],
                    'signal_axon_programs' : genome.function_chromosomes[1],
                    'recieve_axon_signal_programs': genome.function_chromosomes[2],
                    'recieve_reward_programs': genome.function_chromosomes[3],
                    'move_programs': genome.function_chromosomes[4],
                    'die_programs': genome.function_chromosomes[5],
                    'neuron_birth_programs': genome.function_chromosomes[6],
                    'action_controller_programs': genome.function_chromosomes[7],
                    'hox_variant_selection_program': genome.hex_selector_genome.program,
                    'internal_state_variable_count': neuron_internal_states
                }
                axon_init_data = {
                    'recieve_signal_neuron_programs' : genome.function_chromosomes[8],
                    'recieve_signal_dendrite_programs' : genome.function_chromosomes[9],
                    'signal_dendrite_programs' : genome.function_chromosomes[10],
                    'signal_neuron_programs' : genome.function_chromosomes[11],
                    'accept_connection_programs' : genome.function_chromosomes[12],
                    'break_connection_programs' : genome.function_chromosomes[13],
                    'recieve_reward_programs' : genome.function_chromosomes[14],
                    'die_programs' : genome.function_chromosomes[15],
                    'action_controller_programs' : genome.function_chromosomes[16],
                    'internal_state_variable_count': dendrite_internal_states
                }

                return neuron_init_data, axon_init_data

            neuron_init, axon_init = genome_to_init_data(genome)
            engine = NeuronEngine(
                input_arity = problem.input_arity,
                output_arity = problem.output_arity,
                grid_count = 6,
                grid_size = 10,
                actions_max = 120,
                neuron_initialization_data = neuron_init,
                axon_initialization_data = axon_init,
                signal_arity = signal_dimensionality,
                hox_variant_count = hox_variant_count,
                instances_per_iteration = 50,
                logger = logger,
                genome_id = genome.id,
                config_file = copydict(config)
            )

            engine.run(problem, 0)
