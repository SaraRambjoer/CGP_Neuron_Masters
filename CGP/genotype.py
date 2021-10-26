from typing import List
import random
import CGPEngine


FUNCTIONS = ['DUMMYFUNC1', 'DUMMYFUNC2']

class Genome:
    def __init__(self, homeobox_variants, successor_count, input_arities, counter, init_genome = True) -> None:
        # Should hold a HexSelectorGenome, a set of FunctionGenomes, and a ParameterGenome
        self.input_arities = input_arities
        self.counter = counter
        self.homeobox_variants = homeobox_variants
        self.successor_count = successor_count
        if init_genome:
            self.function_chromsomes = []
            for func in FUNCTIONS: 
                self.function_chromsomes.append(FunctionChromosome(homeobox_variants, func, input_arities, counter))
            self.hex_selector_genome = HexSelectorGenome()
            self.parameter_genome = ParameterGenome()
        else:
            self.function_chromsomes = None
            self.hex_selector_genome = None
            self.parameter_genome = None
    
    def mutate(self) -> None: 
        # call crossover stuff in children
        self.hex_selector_genome.mutate()
        self.parameter_genome.mutate()
        for func in self.function_chromsomes:
            func.mutate()

    def crossover(self, target):
        hex_selector_children = self.hex_selector_genome.crossover(target.hex_selector_genome)
        parameter_genome_children = self.parameter_genome.crossover(target.parameter_genome)
        homeobox_variant_children = []
        for num in range(self.homeobox_variants):
            homeobox_variant_children.append(self.function_chromsomes[num].crossover(target.function_chromosomes[num]))

        returned_genomes = []
        for num in range(self.successor_count):
            hex_selector_child = hex_selector_children[num]
            parameter_genome_child = parameter_genome_children[num]
            homeobox_variant_child = homeobox_variant_children[num]
            new_genome = Genome(self.homeobox_variants, self.successor_count, False)
            new_genome.hex_selector_genome = hex_selector_child
            new_genome.parameter_genome = parameter_genome_child
            new_genome.homeobox_variant_child = homeobox_variant_child

        return returned_genomes

# TODO use parameters in parameter genome for controlling this
def generalized_cgp_crossover(evolution_controller, parent1, parent2, child_count):
    program_child_1 = parent1.copy()
    program_child_2 = parent2.copy()
    evolution_controller.crossover(program_child_1, program_child_2)
    children = program_child_1.produce_children(child_count) + program_child_2.produce_children(child_count)
    return children

class HexSelectorGenome:
    def __init__(self,
                 variant_count,
                 internal_state_variables,
                 evolution_controller,
                 program) -> None:
        self.variant_count = variant_count
        self.internal_state_variables = internal_state_variables
        self.input_arity = variant_count + internal_state_variables + 3  # 3 dimensions (x, y, z) input
        self.output_arity = variant_count
        self.evolution_controller = evolution_controller
        self.program = program

    def mutate(self) -> None: 
        # per current design do nothing
        pass

    def crossover(self, other_hexselector, child_count) -> None: 
        # Perform crossover using LEP
        return generalized_cgp_crossover(self.evolution_controller, self, other_hexselector, child_count)


class FunctionChromosome:
    def __init__(self, homeobox_variants, func_name, function_arities, counter) -> None:
        # Should hold a set of homeobox_variants HexFunction variants, should be divided into function types to determine input/output settings, ex. use an enum
        self.homeobox_variants = homeobox_variants
        self.func_name = func_name
        self.hex_variants = []
        self.function_arities = function_arities
        self.counter = counter
        for num in range(homeobox_variants):
            self.hex_variants.append(HexFunction(function_arities[num][0], function_arities[num][1], counter))
    
    def mutate(self) -> None:
        for hex in self.hex_variants:
            hex.mutate()

    def crossover(self, other_chromosome, child_count) -> None:
        # Only supports 2 child crossover
        # Should call crossover operators for homeobox variants where relevant, as well as doing n-point crossover in homeobox-variant space. 
        crossover_point = random.choice(range(0, len(self.hex_variants)))
        child1 = FunctionChromosome()
        child2 = FunctionChromosome()
        child1.hex_variants = self.hex_variants[:crossover_point] + other_chromosome.hex_variants[crossover_point]
        child2.hex_variants = other_chromosome.hex_variants[:crossover_point] + self.hex_variants[crossover_point]
        for x in range(0, len(self.hex_variants)):
            [n1, n2] = child1.hex_variants[x].crossover(child2.hex_variants[x], 2)
            child1.hex_variants[x] = n1
            child2.hex_variants[x] = n2
        return child1, child2

class HexFunction:
    def __init__(self, input_arity, output_arity, counter) -> None:
        self.input_arity = input_arity
        self.output_arity = output_arity
        # Should define a CGP function, should be divided into function types to determine input/output settings
        self.program = CGPEngine.CGPProgram(input_arity, output_arity, counter)

    def mutate(self) -> None:
        pass  # In current design should do nothing

    def crossover(self, other_func, child_count) -> None: 
        return generalized_cgp_crossover(self.evolution_controller, self, other_func, child_count)


class ParameterGenome:
    def __init__(self) -> None:
        # Should consist of all evolvable/adaptive control hyperparameters.
        pass
    
    def mutate(self):
        # Should mutate self according to gaussian distributed around current real number values cropped to range [0, 1]
        pass

    def crossover(self):
        # Should implement some form of n-point crossover
        pass