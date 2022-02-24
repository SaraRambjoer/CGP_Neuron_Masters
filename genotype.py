from typing import List
import CGPEngine
from HelperClasses import randchoice, randcheck, copydict
# TODO child count is misleading

class Genome:
    def __init__(
            self,
            homeobox_variants,
            successor_count,
            input_arities,
            counter, 
            internal_state_variables,
            names,
            logger,
            genome_counter,
            config,
            init_genome = True,
            parent_id = "",
            parent2_id = "",
            hypermutation = False) -> None:
        # Should hold a HexSelectorGenome, a set of FunctionGenomes, and a ParameterGenome
        self.hypermutation = hypermutation
        self.config = copydict(config)
        self.genome_counter = genome_counter
        self.unique_id = str(genome_counter.counterval())
        self.id = f"({parent_id}, {parent2_id}) -> {self.unique_id}"
        self.input_arities = input_arities
        self.counter = counter
        self.homeobox_variants = homeobox_variants
        self.successor_count = successor_count
        self.internal_state_variables = internal_state_variables
        self.names = names
        self.logger = logger
        if init_genome:
            self.function_chromosomes = []
            for num in range(len(input_arities)): 
                name = names[num]
                if name != 'hox_variant_selection_program':
                    self.function_chromosomes.append(FunctionChromosome(homeobox_variants, name, input_arities[num], counter, self.config))
            self.hex_selector_genome = HexSelectorGenome(
                variant_count=homeobox_variants,
                input_arity = input_arities[8][0],
                program = CGPEngine.CGPProgram(input_arities[8][0], input_arities[8][1], counter, self.config),
                config = self.config
            )
            self.parameter_genome = ParameterGenome()
        else:
            self.function_chromosomes = None
            self.hex_selector_genome = None
            self.parameter_genome = None
    
    def update_config(self):
        self.hex_selector_genome.set_config(self.config)
        for funcchrom in self.function_chromosomes:
            funcchrom.set_config(self.config)

    def load(self, sf):
      self.id = sf['genome_id']
      self.unique_id = sf['genome_id'].split("->")[1][1:]
      self.hex_selector_genome.load(sf['hex_selector'])
      for num in range(len(self.function_chromosomes)):
        self.function_chromosomes.load(sf[num+6])


    def mutate(self) -> None: 
        # call crossover stuff in children
        self.hex_selector_genome.mutate()
        self.parameter_genome.mutate()
        for func in self.function_chromosomes:
            func.mutate()
    
    def log(self, initial_data, log=True):
        log_data = {"genome_id" : self.id, "modular_programs" : [], **initial_data}

        def _log_program(program, program_name, log_data):
            active_nodes = program.get_active_nodes() + program.input_nodes
            output_nodes = [(program.nodes[program.output_indexes[x]].id, x) for x in range(len(program.output_indexes))]
            input_nodes = [(program.input_nodes[x].id, x) for x in range(len(program.input_nodes))]
            connection_pairs = []
            for node in active_nodes:
                for input_node in node.inputs:
                    if input_node in active_nodes:
                        connection_pairs.append((input_node.id, node.id, node.inputs.index(input_node)))
                
                if node.gettype()[0:6] == "modular":
                    log_data["modular_programs"].append(_log_program(node.type.program, node.gettype(), {}))

            node_types = [(x.id, x.gettype()) for x in active_nodes]
            active_nodes = [x.id for x in active_nodes]
            log_data[program_name] = {
                "active_nodes" : active_nodes,
                "output_nodes" : output_nodes,
                "input_nodes" : input_nodes,
                "connection_pairs" : connection_pairs,
                "node_types" : node_types
            }

        _log_program(self.hex_selector_genome.program, "hex_selector", log_data)
        for func_chrom in self.function_chromosomes:
            func_name = func_chrom.func_name
            for hexnum in range(len(func_chrom.hex_variants)):
                program = func_chrom.hex_variants[hexnum].program
                _log_program(program, func_name, log_data)

        log_data["adaptive_parameters"] = self.parameter_genome.log()
        if log:
            self.logger.log_json("CGPProgram image", log_data)
        return log_data
    
    def equals_no_id(self, other):
        selflog = self.log({}, False)
        olog = other.log({}, False)
        del selflog['genome_id']
        del olog['genome_id']
        return selflog == olog

    def get_node_type_counts(self):
        node_type_counts = {}
        for key, val in self.hex_selector_genome.program.get_node_type_counts().items():
            if key not in node_type_counts.keys():
                node_type_counts[key] = val
            else:
                node_type_counts[key] += val
        for cgp_function_chromsome in self.function_chromosomes:
            for hex_variant in cgp_function_chromsome.hex_variants:
                for key, val in hex_variant.program.get_node_type_counts().items():
                    if key not in node_type_counts.keys():
                        node_type_counts[key] = val
                    else:
                        node_type_counts[key] += val
        return node_type_counts
        
    def add_cgp_modules_to_list(self, input_list, genome, recursive=False):
        maxdepth = 0
        for cgp_module, depth in genome.hex_selector_genome.program.get_active_modules(recursive):
            maxdepth = max(maxdepth, depth)
            input_list.append(cgp_module)
        for cgp_function_chromosome in genome.function_chromosomes:
            for hex_variant in cgp_function_chromosome.hex_variants:
                for cgp_module, depth in hex_variant.program.get_active_modules(recursive):
                    maxdepth = max(maxdepth, depth)
                    input_list.append(cgp_module)
        return input_list, maxdepth

    def crossover(self, target):
        self.update_config()
        
        cgp_modules = []
        # Has the side effect of favouring commonly used modules more heavily as they should appear more often in the list
        # Which seems like reasonable behaviour
        self.add_cgp_modules_to_list(cgp_modules, self)
        self.add_cgp_modules_to_list(cgp_modules, target)
        hex_selector_children = self.hex_selector_genome.crossover(target.hex_selector_genome, self.successor_count, cgp_modules)
        parameter_genome_children = self.parameter_genome.crossover(target.parameter_genome, self.successor_count)
        function_chromosome_children = []
        for num in range(len(self.function_chromosomes)):
            function_chromosome_children.append(self.function_chromosomes[num].crossover(target.function_chromosomes[num], 2, cgp_modules))

        returned_genomes = []
        for num in range(self.successor_count):
            hex_selector_child = hex_selector_children[num]
            parameter_genome_child = parameter_genome_children[num]
            function_chromosome_child = [x[num] for x in function_chromosome_children]
            new_genome = Genome(
                homeobox_variants = self.homeobox_variants, 
                successor_count = self.successor_count, 
                input_arities = self.input_arities,
                counter = self.counter,
                internal_state_variables = self.internal_state_variables,
                names = self.names,
                logger = self.logger,
                genome_counter = self.genome_counter,
                config = self.config,
                init_genome = False,
                parent_id = self.unique_id,
                parent2_id = target.unique_id,
                hypermutation=self.hypermutation)
            new_genome.hex_selector_genome = hex_selector_child
            new_genome.parameter_genome = parameter_genome_child
            new_genome.function_chromosomes = function_chromosome_child
            new_genome.update_config()
            returned_genomes.append(new_genome)

        return returned_genomes

    def __eq__(self, o: object) -> bool:
        if type(o) != Genome:
            return False
        selflog = self.log({}, False)
        olog = o.log({}, False)
        return selflog == olog
    

    def __str__(self) -> str:
        return str(self.hex_selector_genome) + "\n----\n" + "\n-----\n".join([str(x) for x in self.function_chromosomes])

# TODO use parameters in parameter genome for controlling this
def generalized_cgp_crossover(parent1, parent2, child_count, samemut, cgp_modules = None):
    if samemut:
        program_child_1 = parent1.program.deepcopy()
        program_child_2 = parent2.program.deepcopy()
        program_child_3 = parent1.program.deepcopy()
        program_child_4 = parent2.program.deepcopy()

        program_child_1.config['mutation_chance_node'] = parent1.config['mutation_chance_node']
        program_child_2.config['mutation_chance_node'] = parent1.config['mutation_chance_node']
        program_child_3.config['mutation_chance_node'] = parent1.config['mutation_chance_node']
        program_child_4.config['mutation_chance_node'] = parent1.config['mutation_chance_node']

        CGPEngine.subgraph_crossover(program_child_1, program_child_2, 12, 12)
        children = program_child_1.produce_children(1, cgp_modules) + program_child_2.produce_children(1, cgp_modules) + \
          program_child_3.produce_children(1, cgp_modules) + program_child_4.produce_children(1, cgp_modules)
        return children
    else:
        program_child_1 = parent1.program.deepcopy()
        program_child_2 = parent2.program.deepcopy()

        program_child_1.config['mutation_chance_node'] = parent1.config['mutation_chance_node']
        program_child_2.config['mutation_chance_node'] = parent1.config['mutation_chance_node']

        CGPEngine.subgraph_crossover(program_child_1, program_child_2, 12, 12)
        children = program_child_1.produce_children(1, cgp_modules) + program_child_2.produce_children(1, cgp_modules)
        return children


class HexSelectorGenome:
    def __init__(self,
                 variant_count,
                 input_arity,
                 program,
                 config) -> None:
        self.variant_count = variant_count
        self.input_arity = input_arity
        self.output_arity = variant_count
        self.program = program
        self.config = config

    def mutate(self) -> None: 
        # per current design do nothing
        pass

    def set_config(self, config):
        self.config = config
        self.program.set_config(config)

    def crossover(self, other_hexselector, child_count, cgp_modules= None): 
        children = generalized_cgp_crossover(self, other_hexselector, child_count, self.config['non_crossover_children'], cgp_modules)
        outputs = []
        for child in children:
            outputs.append(
                HexSelectorGenome(
                    variant_count = self.variant_count,
                    input_arity = self.input_arity,
                    program = child,
                    config = self.config
                )
            )
        return outputs
    
    def __eq__(self, o: object) -> bool:
        return self.program == o.program

    def __str__(self) -> str:
        return str(self.program)


class FunctionChromosome:
    def __init__(self, homeobox_variants, func_name, function_arities, counter, config) -> None:
        # Should hold a set of homeobox_variants HexFunction variants, should be divided into function types to determine input/output settings, ex. use an enum
        self.homeobox_variants = homeobox_variants
        self.func_name = func_name
        self.hex_variants = []
        self.function_arities = function_arities
        self.counter = counter
        self.config = config
        for num in range(homeobox_variants):
            self.hex_variants.append(HexFunction(function_arities[0], function_arities[1], counter, self.config))
    
    def set_config(self, config):
        self.config = config
        for hexfunc in self.hex_variants:
            hexfunc.set_config(config)

    def mutate(self) -> None:
        for hex in self.hex_variants:
            hex.mutate()

    def crossover(self, other_chromosome, child_count, cgp_modules = None) -> None:
        # Should call crossover operators for homeobox variants where relevant, as well as doing n-point crossover in homeobox-variant space. 
        child1 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter, self.config)
        child2 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter, self.config)
        if self.config['non_crossover_children']:
            child3 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter, self.config)
            child4 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter, self.config)
            child3.hex_variants = self.hex_variants
            child4.hex_variants = other_chromosome.hex_variants

        for num in range(len(self.hex_variants)):
            if randcheck(self.config['hex_crossover_chance']):
                child1.hex_variants[num] = other_chromosome.hex_variants[num]
            else: 
                child1.hex_variants[num] = self.hex_variants[num]

        for num in range(len(self.hex_variants)):
            if randcheck(self.config['hex_crossover_chance']):
                child2.hex_variants[num] = other_chromosome.hex_variants[num]
            else: 
                child2.hex_variants[num] = self.hex_variants[num]

        for x in range(0, len(self.hex_variants)):
            if self.config['non_crossover_children']:
                [n1, n2, n3, n4] = child1.hex_variants[x].crossover(child2.hex_variants[x], 2, cgp_modules)
                child1.hex_variants[x].program = n1
                child2.hex_variants[x].program = n2
                child3.hex_variants[x].program = n3
                child4.hex_variants[x].program = n4
            else:
                [n1, n2] = child1.hex_variants[x].crossover(child2.hex_variants[x], 2, cgp_modules)
                child1.hex_variants[x].program = n1
                child2.hex_variants[x].program = n2
        if self.config['non_crossover_children']:
            return child1, child2, child3, child4
        else:
            return child1, child2
    
    def __eq__(self, o: object) -> bool:
        equa = True
        for num in range(len(self.hex_variants)):
            equa = equa and self.hex_variants[num] == o.hex_variants[num]
    
    def __str__(self) -> str:
        return self.func_name + "\n" + "\n".join([str(x) for x in self.hex_variants])

class HexFunction:
    def __init__(self, input_arity, output_arity, counter, config) -> None:
        self.input_arity = input_arity
        self.config = config
        self.output_arity = output_arity
        # Should define a CGP function, should be divided into function types to determine input/output settings
        self.program = CGPEngine.CGPProgram(input_arity, output_arity, counter, self.config)

    def set_config(self, config):
        self.config = config
        self.program.set_config(config)

    def mutate(self) -> None:
        pass  # In current design should do nothing

    def crossover(self, other_func, child_count, cgp_modules = None) -> None: 
        return generalized_cgp_crossover(self, other_func, child_count, self.config['non_crossover_children'], cgp_modules)
    
    def __eq__(self, o: object) -> bool:
        return self.program == o.program

    def __str__(self) -> str:
        return str(self.program)


class ParameterGenome:
    def __init__(self) -> None:
        # Should consist of all evolvable/adaptive control hyperparameters.
        pass
    
    def mutate(self):
        # Should mutate self according to gaussian distributed around current real number values cropped to range [0, 1]
        pass

    def crossover(self, other_parameter_genome, child_count):
        # Should implement some form of n-point crossover
        return [ParameterGenome() for _ in range(child_count)]
    
    def log(self):
        # should return log describing values as a dictionary
        return "not implemented"