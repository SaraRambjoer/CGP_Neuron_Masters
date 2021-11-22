from CGPEngine import * 
import random
import numpy as np

def addition_problem():
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    term1 = random.choice(nums)
    term2 = random.choice(nums)
    result = term1 + term2
    
    program = CGPProgram(2, 1)
    print(program.run((term1, term2))[0] == result)

#for num in range(0, 100):
#    addition_problem()

def addition_problem_evolutionary():
    inputs = []
    target_outputs = []
    for num in range(100):
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        term1 = random.choice(nums)
        term2 = random.choice(nums)
        result = term1 + term2 + term2 * term2
        inputs.append((term1, term2))
        target_outputs.append(result)
    counter = Counter()
    program = CGPProgram(2, 1, counter)
    targets = np.asarray(target_outputs)
    def _eval_routine(predictions):
        preds = np.asarray(predictions).flatten()
        diff = np.sum(np.abs(targets-preds))

        to_return = (np.sum(targets) - diff)/np.sum(targets)
        return to_return
    for num in range(2000):
        program.four_plus_one_evolution(lambda x: _eval_routine(x), inputs)


#addition_problem_evolutionary()


def addition_problem_parallell_evolutionary():
    inputs = []
    target_outputs = []
    for num in range(100):
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        term1 = random.choice(nums)
        term2 = random.choice(nums)
        result = term1 + term2 + term2 * term2
        inputs.append((term1, term2))
        target_outputs.append(result)
    targets = np.asarray(target_outputs)
    evolution_controller = EvolutionController(40, 3, 2, 1, True, 70, 12, 12)
    def _eval_routine(predictions):
        preds = np.asarray(predictions).flatten()
        diff = np.sum(np.abs(targets-preds))

        to_return = (np.sum(targets) - diff)/np.sum(targets)
        return to_return
    for num in range(2000):
        evolution_controller.evolution_step(inputs, lambda x: _eval_routine(x))

addition_problem_parallell_evolutionary()