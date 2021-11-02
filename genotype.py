from typing import List
import CGPEngine
from HelperClasses import randchoice
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
            init_genome = True) -> None:
        # Should hold a HexSelectorGenome, a set of FunctionGenomes, and a ParameterGenome
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
                    self.function_chromosomes.append(FunctionChromosome(homeobox_variants, name, input_arities[num], counter))
            self.hex_selector_genome = HexSelectorGenome(
                variant_count=homeobox_variants,
                input_arity = input_arities[7][0],
                program = CGPEngine.CGPProgram(input_arities[7][0], input_arities[7][1], counter)
            )
            self.parameter_genome = ParameterGenome()
        else:
            self.function_chromosomes = None
            self.hex_selector_genome = None
            self.parameter_genome = None
    
    def mutate(self) -> None: 
        # call crossover stuff in children
        self.hex_selector_genome.mutate()
        self.parameter_genome.mutate()
        for func in self.function_chromosomes:
            func.mutate()

    def crossover(self, target):
        hex_selector_children = self.hex_selector_genome.crossover(target.hex_selector_genome, 2)
        parameter_genome_children = self.parameter_genome.crossover(target.parameter_genome, 2)
        function_chromosome_children = []
        for num in range(len(self.function_chromosomes)):
            # BUG Something weird with creating too many hex variants?
            function_chromosome_children.append(self.function_chromosomes[num].crossover(target.function_chromosomes[num], 2))
        for chrom in function_chromosome_children:
            chro = chrom[0]
            self.logger.log("CGPProgram image", chro.func_name + "\n")
            for num in range(self.homeobox_variants):
                self.logger.log("CGPProgram image", "Hox variant " + str(num) + "\n")
                program = chro.hex_variants[num].program
                output_nodes = [program.nodes[x] for x in program.output_indexes]
                self.logger.log_cgp_program(
                    program.get_active_nodes(),
                    output_nodes
                )

        returned_genomes = []
        for num in range(self.successor_count):
            hex_selector_child = hex_selector_children[num]
            parameter_genome_child = parameter_genome_children[num]
            function_chromosome_child = [x[num] for x in function_chromosome_children]
            new_genome = Genome(
                self.homeobox_variants, 
                self.successor_count, 
                self.input_arities,
                self.counter,
                self.internal_state_variables,
                self.names,
                self.logger,
                False)
            new_genome.hex_selector_genome = hex_selector_child
            new_genome.parameter_genome = parameter_genome_child
            new_genome.function_chromosomes = function_chromosome_child
            returned_genomes.append(new_genome)

        return returned_genomes
    def __eq__(self, o: object) -> bool:
        equa = self.hex_selector_genome == o.hex_selector_genome
        for num in range(len(self.function_chromosomes)):
            equa = equa and self.function_chromosomes[num] == o.function_chromosomes[num]
    

    def __str__(self) -> str:
        return str(self.hex_selector_genome) + "\n----\n" + "\n-----\n".join([str(x) for x in self.function_chromosomes])

# TODO use parameters in parameter genome for controlling this
def generalized_cgp_crossover(parent1, parent2, child_count):
    program_child_1 = parent1.program.deepcopy()
    program_child_2 = parent2.program.deepcopy()
    CGPEngine.subgraph_crossover(program_child_1, program_child_2, 12, 12)
    children = program_child_1.produce_children(1) + program_child_2.produce_children(1)
    return children

class HexSelectorGenome:
    def __init__(self,
                 variant_count,
                 input_arity,
                 program) -> None:
        self.variant_count = variant_count
        self.input_arity = input_arity
        self.output_arity = variant_count
        self.program = program

    def mutate(self) -> None: 
        # per current design do nothing
        pass

    def crossover(self, other_hexselector, child_count) -> None: 
        # Perform crossover using LEP
        children = generalized_cgp_crossover(self, other_hexselector, child_count)
        outputs = []
        for child in children:
            outputs.append(
                HexSelectorGenome(
                    self.variant_count,
                    self.input_arity,
                    child
                )
            )
        return outputs
    
    def __eq__(self, o: object) -> bool:
        return self.program == o.program

    def __str__(self) -> str:
        return str(self.program)


class FunctionChromosome:
    def __init__(self, homeobox_variants, func_name, function_arities, counter) -> None:
        # Should hold a set of homeobox_variants HexFunction variants, should be divided into function types to determine input/output settings, ex. use an enum
        self.homeobox_variants = homeobox_variants
        self.func_name = func_name
        self.hex_variants = []
        self.function_arities = function_arities
        self.counter = counter
        for num in range(homeobox_variants):
            self.hex_variants.append(HexFunction(function_arities[0], function_arities[1], counter))
    
    def mutate(self) -> None:
        for hex in self.hex_variants:
            hex.mutate()

    def crossover(self, other_chromosome, child_count) -> None:
        # Only supports 2 child crossover
        # Should call crossover operators for homeobox variants where relevant, as well as doing n-point crossover in homeobox-variant space. 
        crossover_point = randchoice(range(0, len(self.hex_variants)))
        child1 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter)
        child2 = FunctionChromosome(self.homeobox_variants, self.func_name, self.function_arities, self.counter)
        child1.hex_variants = self.hex_variants[:crossover_point] + other_chromosome.hex_variants[crossover_point:]
        child2.hex_variants = other_chromosome.hex_variants[:crossover_point] + self.hex_variants[crossover_point:]
        for x in range(0, len(self.hex_variants)):
            [n1, n2] = child1.hex_variants[x].crossover(child2.hex_variants[x], 2)
            child1.hex_variants[x].program = n1
            child2.hex_variants[x].program = n2
        return child1, child2
    
    def __eq__(self, o: object) -> bool:
        equa = True
        for num in range(len(self.hex_variants)):
            equa = equa and self.hex_variants[num] == o.hex_variants[num]
    
    def __str__(self) -> str:
        return self.func_name + "\n" + "\n".join([str(x) for x in self.hex_variants])

class HexFunction:
    def __init__(self, input_arity, output_arity, counter) -> None:
        self.input_arity = input_arity
        self.output_arity = output_arity
        # Should define a CGP function, should be divided into function types to determine input/output settings
        self.program = CGPEngine.CGPProgram(input_arity, output_arity, counter)

    def mutate(self) -> None:
        pass  # In current design should do nothing

    def crossover(self, other_func, child_count) -> None: 
        return generalized_cgp_crossover(self, other_func, child_count)
    
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
        return [ParameterGenome(), ParameterGenome()]