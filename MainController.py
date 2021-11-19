def log_genome(genomes, runinfo):
    for genome in genomes:
        initial_data = {
            "genome id" : genome[0].id,
            "genome fitness" : genome[1],
            "run" : runinfo
        }
        genome[0].log(initial_data)

if __name__ == "__main__":
    # Setup problems
    from genotype import Genome
    import Logger
    import stupid_problem_test
    from pathos.multiprocessing import Pool
    import random
    problem = stupid_problem_test.StupidProblem()
    # Setup logging
    # ["CGPProgram image", "cgp_function_exec_prio1", "cgp_function_exec_prio2", "graphlog_instance", "graphlog_run", "setup_info"]
    logger = Logger.Logger("C:/users/jonod/desktop/masters/logfiles/log", ["cgp_function_exec_prio1", "cgp_function_exec_prio2", "graphlog_instance"])
    # Setup CGP genome
    # - define a counter
    from HelperClasses import Counter, randchoice, drawProgram
    counter = Counter()
    neuron_internal_states = 1
    dendrite_internal_states = 1
    signal_dimensionality = 1
    dimensions = 3  # other dimensions not supported - code in engine.py specific to 3d grid
    hox_variant_count = 1
    genome_counter = Counter()
    genome_count = 40
    seed = 200
    random.seed(seed)
    
    logger.log_json("setup_info", {
        "neuron internal state count" : neuron_internal_states,
        "axon-dendrite internal state count" : dendrite_internal_states,
        "signal dimensionality" : signal_dimensionality,
        "hox_variant_count" : hox_variant_count,
        "genome_count" : genome_count,
        "random_seed" : seed
    })

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

    import CGPEngine
    # initialize the genome(s)
    all_function_arities = neuron_function_arities + dendrite_function_arities
    genomes = []
    for num in range(genome_count):
        genomes.append(Genome(
            hox_variant_count,
            2,
            all_function_arities,
            counter,
            neuron_internal_states,
            neuron_function_order[:-1] + dendrite_function_order,
            logger,
            genome_counter)) # TODO RN assumes equal amount of axon and neuron internal state variables


    from engine import NeuronEngine
    # learning loop
    learning_iterations = 1000
    genome_results = []
    neuron_init, axon_init = genome_to_init_data(genomes[0])
    for genome in genomes:
        engine = NeuronEngine(
            input_arity = problem.input_arity,
            output_arity = problem.output_arity,
            grid_count = 6,
            grid_size = 10,
            actions_max = 70,
            neuron_initialization_data = neuron_init,
            axon_initialization_data = axon_init,
            signal_arity = signal_dimensionality,
            hox_variant_count = hox_variant_count,
            instances_per_iteration = 50,
            logger = logger
        )
        result, base_problems = engine.run(problem, "setup")
        genome_results.append((result, base_problems))
    genomes = list(zip(genomes, [x[0] for x in genome_results], [x[1] for x in genome_results]))
    log_genome(genomes, 0)
    for num in range(learning_iterations):    
        new_genomes = []
        egligable_bachelors = [x[0] for x in genomes]
        while len(egligable_bachelors) > 0:
            choice1 = randchoice(egligable_bachelors)
            choice2 = randchoice(egligable_bachelors)
            indexes = [egligable_bachelors.index(choice1), egligable_bachelors.index(choice2)]
            egligable_bachelors.remove(choice1)  # Currently possible to do crossover with self, which does make some sense with subgraph extraction
            if choice2 in egligable_bachelors:
                egligable_bachelors.remove(choice2)
            new_genomes = choice1.crossover(choice2)[:2] # remove [:2] to not only get crossover children
            genome_results = []
            engines = []
            for genome in new_genomes:
                neuron_initialization_data, axon_initialization_data = genome_to_init_data(genome)
                engine = NeuronEngine(
                    input_arity = problem.input_arity,
                    output_arity = problem.output_arity,
                    grid_count = 6,
                    grid_size = 10,
                    actions_max = 70,
                    neuron_initialization_data = neuron_initialization_data,
                    axon_initialization_data = axon_initialization_data,
                    signal_arity = signal_dimensionality,
                    hox_variant_count = hox_variant_count,
                    instances_per_iteration = 50,
                    logger = logger
                )
                engines.append(engine)

            #def multiprocess_code(engine_problem):
            #    return engine_problem[0].run(engine_problem[1])
          #
            #with Pool() as p:
            #    results = p.map(multiprocess_code, list(zip(new_genomes, [stupid_problem_test.StupidProblem() for _ in range(len(new_genomes))])))
            genome_results = [engine.run(problem, num) for engine in engines]
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

            parent1score = genomes[indexes[0]][1]
            parent2score = genomes[indexes[1]][1]
            for num2 in range(len(new_genomes)):
                # SHOULDN'T HAPPEN
                if new_genomes[num2] == genomes[indexes[0]][0]:
                    print("EQUALS")
                elif new_genomes[num2] == genomes[indexes[1]][0]:
                    print("EQUALS2")
                # SHOULDN'T HAPPEN (strictly speaking some chance tho)

                if genome_results[num2] <= parent1score:
                    #_draw_program_data(genomes[indexes[0]][0])
                    #_draw_program_data(new_genomes[num2])
                    genomes[indexes[0]] = (new_genomes[num2], genome_results[num2], base_problems[num2])
                    parent1score = genome_results[num2] 
                if genome_results[num2] <= parent2score:
                    genomes[indexes[1]] = (new_genomes[num2], genome_results[num2], base_problems[num2])
                    parent2score = genome_results[num2]
        #print(num, [f"{x[1]}, {x[2]}" for x in genomes])
        print(f"------------------- {num} ------------------")
        for genome in genomes: 
            print("\n\n")
            print(genome[1])
            print(genome[2])
        log_genome(genomes, num)
        #_genomes = [x[0] for x in genomes]
        #for gen in _genomes:
        #  print(str(gen))
      
