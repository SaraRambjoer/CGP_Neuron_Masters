from typing import List


FUNCTIONS = ['DUMMYFUNC1', 'DUMMYFUNC2']

class Genome:
    def __init__(self, homeobox_variants, successor_count, init_genome = True) -> None:
        # Should hold a HexSelectorGenome, a set of FunctionGenomes, and a ParameterGenome
        self.homeobox_variants = homeobox_variants
        self.successor_count = successor_count
        if init_genome:
            self.function_chromsomes = []
            for func in FUNCTIONS: 
                self.function_chromsomes.append(FunctionChromosome(homeobox_variants, func))
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

    def crossover(self, target: Genome) -> List[Genome]:
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




class HexSelectorGenome:
    def __init__(self) -> None:
        pass

    def mutate(self) -> None: 
        # per current design do nothing
        pass

    def crossover(self) -> None: 
        # Perform crossover using LEP
        pass


class FunctionChromosome:
    def __init__(self, homeobox_variants, func_name) -> None:
        # Should hold a set of homeobox_variants HexFunction variants, should be divided into function types to determine input/output settings, ex. use an enum
        self.homeobox_variants = homeobox_variants
        self.func_name = func_name
        self.hex_variants = []
        for num in range(homeobox_variants):
            self.hex_variants.append(HexFunction())
    
    def mutate(self) -> None:
        for hex in self.hex_variants:
            hex.mutate()

    def crossover(self) -> None:
        # Should call crossover operators for homeobox variants where relevant, as well as doing n-point crossover in homeobox-variant space. 
        pass


class HexFunction:
    def __init__(self) -> None:
        # Should define a CGP function, should be divided into function types to determine input/output settings
        pass

    def mutate(self) -> None:
        pass  # In current design should do nothing

    def crossover(self) -> None: 
        # should perform crossover using Local Evolution Pool
        pass


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