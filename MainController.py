import time
import json

from numpy import diag
from engine import NeuronEngine
from genotype import Genome
import Logger
import stupid_problem_test
import random
from HelperClasses import Counter, randchoice, copydict, randcheck, copydict
import os
from multiprocessing import Pool

def multiprocess_code(engine_problem):
    engine = engine_problem[0]
    problem = engine_problem[1]
    num = engine_problem[2]
    to_return = engine.run(problem, num)
    return to_return

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
            if config[key][0].isnumeric():
                for num2 in range(len(config[key])):
                    config[key][num2] = int(config[key][num2])

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
    logger = Logger.Logger(os.path.join(os.path.dirname(__file__), "logfiles") + "\\log", config['logger_ignore_messages'], config['advanced_logging'])
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

    estimated_calls = 1.1*config['genome_count']*config['iterations']*config['cgp_program_size']*config['actions_max']*2*config['instances_per_iteration']
    print(f"Estimated upper limit to calls to CGP node primitives: {estimated_calls}")
    print(f"Estimated total computation time at upper limit: {500*estimated_calls/1600000} seconds")
    print(f"Based on limited empirical data actual computation time will often be up to 70 times as low.")

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
        [dimensions+neuron_internal_states+1 + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+neuron_internal_states],  # axon birth
        [signal_dimensionality+dimensions+neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + signal_dimensionality + neuron_internal_states],  # signal axon
        [signal_dimensionality + dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + neuron_internal_states+signal_dimensionality],  # recieve signal axon
        [1 + dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + neuron_internal_states],  # reciee reward
        [neuron_internal_states + dimensions + len(config['cgp_function_constant_numbers']), 7+neuron_internal_states],  # move
        [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2+neuron_internal_states],  # die
        [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2+neuron_internal_states*2],  # neuron birth
        [neuron_internal_states+dimensions + len(config['cgp_function_constant_numbers']), 9],  # action controller
        [neuron_internal_states + dimensions + len(config['cgp_function_constant_numbers']), hox_variant_count]  # hox selection
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
        [dendrite_internal_states + signal_dimensionality + dimensions + len(config['cgp_function_constant_numbers']), 2+signal_dimensionality+dendrite_internal_states],
        [dendrite_internal_states + signal_dimensionality + dimensions + len(config['cgp_function_constant_numbers']), 2+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + signal_dimensionality + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + signal_dimensionality + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+dendrite_internal_states],
        [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 2+dendrite_internal_states], # Accept connection
        [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 1], # Break connection
        [dimensions + dendrite_internal_states + 1 + len(config['cgp_function_constant_numbers']), 2 + dendrite_internal_states], # recieve reward
        [dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 1+signal_dimensionality], # die
        [dendrite_internal_states + dimensions + len(config['cgp_function_constant_numbers']), 3]
    ]

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

    grid_count = config['grid_count']
    grid_size = config['grid_size']
    actions_max = config['actions_max']
    instances_per_iteration = config['instances_per_iteration']

    genome_results = []
    neuron_init, axon_init = genome_to_init_data(genomes[0])
    for genome in genomes:
        engine = NeuronEngine(
            input_arity = problem.input_arity,
            output_arity = problem.output_arity,
            grid_count = grid_count,
            grid_size = grid_size,
            actions_max = actions_max,
            neuron_initialization_data = neuron_init,
            axon_initialization_data = axon_init,
            signal_arity = signal_dimensionality,
            hox_variant_count = hox_variant_count,
            instances_per_iteration = instances_per_iteration,
            logger = logger,
            genome_id = genome.id,
            config_file = copydict(config)
        )
        result, base_problems = engine.run(problem, "setup")
        genome_results.append((result, base_problems))
    genomes = list(zip(genomes, [x[0] for x in genome_results], [x[1] for x in genome_results]))
    to_return_fitness.append(x[0] for x in genome_results)
    log_genome(genomes, 0)

    diagnostic_data = {}
    diagnostic_data['config'] = copydict(config)
    diagnostic_data['iterations'] = []

    print("Setup complete. Beginning evolution.")

    for num in range(learning_iterations):   
        statistic_entry = {}

        time_genes = 0
        time_eval = 0
        time_genes_post = 0
        time_genes_selection = 0
        time_genes_crossover = 0
        time_genes_skip_check = 0
        egligable_bachelors = [x[0] for x in genomes]
        child_data = [[] for _ in range(len(genomes))]
        while len([x for x in egligable_bachelors if x is not None]) > 0:
            time_genes_stamp = time.time()
            time_genes_selection_stamp = time.time()
            choice1 = randchoice([x for x in egligable_bachelors if x is not None])
            choice2 = randchoice([x for x in egligable_bachelors if x is not None])
            indexes = [egligable_bachelors.index(choice1), egligable_bachelors.index(choice2)]
            egligable_bachelors[egligable_bachelors.index(choice1)] = None  # Currently possible to do crossover with self, which does make some sense with subgraph extraction
            if choice2 in egligable_bachelors and choice2 != choice1:
                egligable_bachelors[egligable_bachelors.index(choice2)] = None
            time_genes_selection += time.time() - time_genes_selection_stamp
            time_genes_crossover_stamp = time.time()
            new_genomes = choice1.crossover(choice2)
            time_genes_crossover += time.time() - time_genes_crossover_stamp
            time_genes_skip_check_stamp = time.time()
            skip_eval = [False for num in range(len(new_genomes))]
            for numero in range(len(new_genomes)):
                genome = new_genomes[numero]
                if genome.equals_no_id(choice1):
                    skip_eval[numero] = 1
                if genome.equals_no_id(choice2):
                    skip_eval[numero] = 2
            time_genes_skip_check += time.time() - time_genes_skip_check_stamp
            genome_results = []
            time_genes += time.time() - time_genes_stamp
            time_eval_stamp = time.time()


            engine_problems = []
            for numero in range(len(new_genomes)):
                genome = new_genomes[numero]
                if not skip_eval[numero]:
                    neuron_initialization_data, axon_initialization_data = genome_to_init_data(genome)
                    engine = NeuronEngine(
                        input_arity = problem.input_arity,
                        output_arity = problem.output_arity,
                        grid_count = grid_count,
                        grid_size = grid_size,
                        actions_max = actions_max,
                        neuron_initialization_data = neuron_initialization_data,
                        axon_initialization_data = axon_initialization_data,
                        signal_arity = signal_dimensionality,
                        hox_variant_count = hox_variant_count,
                        instances_per_iteration = instances_per_iteration,
                        logger = logger,
                        genome_id = genome.id,
                        config_file = copydict(config)
                    )
                    engine_problems.append((engine, problem, num))
                elif skip_eval[numero] == 1:
                    genome_results.append((genomes[indexes[0]][1], genomes[indexes[0]][2]))
                else:
                    genome_results.append((genomes[indexes[1]][1], genomes[indexes[1]][2]))

            with Pool() as p:
                results = p.map(multiprocess_code, engine_problems)
            
            genome_results += results
        
            #for numero in range(len(new_genomes)):
            #    genome = new_genomes[numero]
            #    if not skip_eval[numero]:
            #        neuron_initialization_data, axon_initialization_data = genome_to_init_data(genome)
            #        engine = NeuronEngine(
            #            input_arity = problem.input_arity,
            #            output_arity = problem.output_arity,
            #            grid_count = grid_count,
            #            grid_size = grid_size,
            #            actions_max = actions_max,
            #            neuron_initialization_data = neuron_initialization_data,
            #            axon_initialization_data = axon_initialization_data,
            #            signal_arity = signal_dimensionality,
            #            hox_variant_count = hox_variant_count,
            #            instances_per_iteration = instances_per_iteration,
            #            logger = logger,
            #            genome_id = genome.id,
            #            config_file = copydict(config)
            #        )
            #        genome_results.append(engine.run(problem, num))
            #    elif skip_eval[numero] == 1:
            #        genome_results.append((genomes[indexes[0]][1], genomes[indexes[0]][2]))
            #    else:
            #        genome_results.append((genomes[indexes[1]][1], genomes[indexes[1]][2]))

            time_eval += time.time() - time_eval_stamp


            time_genes_stamp = time.time()
            base_problems = [x[1] for x in genome_results]
            genome_results = [x[0] for x in genome_results]
            # all children of a parent compete for the parents spots

            for x in range(len(new_genomes)):
                child_data[indexes[0]].append((new_genomes[x], genome_results[x], base_problems[x]))
                child_data[indexes[1]].append((new_genomes[x], genome_results[x], base_problems[x]))        
            time_genes += time.time() - time_genes_stamp

        time_genes_post_stamp = time.time()
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
        
        times_a_genome_took_population_slot_from_other_genome = 0
        average_takeover_probability = 0
        for num4 in range(config['genome_replacement_tries']):
            genome_one = randchoice(genomes)
            genome_two = randchoice([x for x in genomes if x is not genome_one])
            diff = abs(genome_one[1] - genome_two[1])
            maxi = max(genome_one[1], genome_two[1])
            average_takeover_probability += diff*config['replacement_fitness_difference_scaling']/maxi
            if diff > config['replacement_fitness_difference_threshold']:
                if randcheck(diff*config['replacement_fitness_difference_scaling']/maxi):
                    if genome_one[1] > genome_two[1]:
                        genomes[genomes.index(genome_two)] = genome_one
                    else:
                        genomes[genomes.index(genome_one)] = genome_two
                    times_a_genome_took_population_slot_from_other_genome += 1
        
        if times_a_genome_took_population_slot_from_other_genome != 0:
            average_takeover_probability = average_takeover_probability/config['genome_replacement_tries']
        statistic_entry["genome_replacement_stats"] = {
            "times_a_genome_took_population_slot_from_other_genome" : times_a_genome_took_population_slot_from_other_genome,
            "average_takover_probability" : average_takeover_probability
        }



        time_genes_post += time.time() - time_genes_post_stamp
        #print(num, [f"{x[1]}, {x[2]}" for x in genomes])
        print(f"------------------- {num} ------------------")
        print(f"genes:{time_genes}, genes_selection:{time_genes_selection}, genes_crossover:{time_genes_crossover}, " +\
            f"genes_skip_check:{time_genes_skip_check}, eval:{time_eval}, genes_post:{time_genes_post}")
        
        statistic_entry['iteration'] = num

        time_statistic_entry = {
            "genes":time_genes,
            "genes_selection":time_genes_selection,
            "genes_crossover":time_genes_crossover,
            "genes_skip_check":time_genes_skip_check,
            "eval":time_eval,
            "genes_post":time_genes_post
        }
        statistic_entry['time'] = time_statistic_entry
        
        
        genomes_data = {
            "genome_list":[]
        }
        for genome in genomes: 

            module_list, _ = genome[0].add_cgp_modules_to_list([], genome[0])
            module_list_recursive, module_max_depth = genome[0].add_cgp_modules_to_list([], genome[0], True)
            node_type_counts = genome[0].get_node_type_counts()
            total_active_nodes = sum(node_type_counts.values())
            genome_entry = {
                "id":genome[0].id,
                "fitness":genome[1],
                "performance_stats":genome[2],
                "node_mutation_chance":genome[0].config['mutation_chance_node'],
                "link_mutation_chance":genome[0].config['mutation_chance_link'],
                "module_count_non_recursive":len(module_list),
                "module_count_recursive":len(module_list_recursive),
                "cgp_node_types": node_type_counts
            }
            genomes_data["genome_list"] += [genome_entry]
            print()
            print(genome[0].id)
            print(genome[1])
            print(genome[2])
            print(genome[0].config['mutation_chance_node'], genome[0].config['mutation_chance_link'])
        
        statistic_entry['genomes_data'] = genomes_data
        diagnostic_data['iterations'] += [statistic_entry]

        to_return_fitness.append([x[1] for x in genomes])
        log_genome(genomes, num)
        #_genomes = [x[0] for x in genomes]
        #for gen in _genomes:
        #  print(str(gen))
    logger.log_statistic_data(diagnostic_data)
    return to_return_fitness, diagnostic_data

if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config["Default"]
    config = process_config(config)
    if config['mode'] == 'run':
        print("Running evolution")
        import cProfile
        cProfile.run("run(config, print_output=True)")
        #run(config, print_output=True)
    elif config['mode'][0] == 'load':
        # TODO not fully implemented
        # TODO if fully implementing unify code with run function better, outdated due to code duplications
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
            logger = Logger.Logger(os.path.join(os.path.dirname(__file__), "logfiles") + "\\log", config['logger_ignore_messages'], config['advanced_logging'])
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
