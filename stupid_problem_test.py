from HelperClasses import randchoice

input_arity = 4
output_arity = 1

true_samples = [
    '0000',
    '0001',
    '0010',
    '0011',
    '0100',
    '0101',
    '0110',
    '0111'
]

false_samples = [
    '1000',
    '1001',
    '1010',
    '1011',
    '1100',
    '1101',
    '1110',
    '1111'
]

def str_to_list(inputstr):
    outputlist = []
    for character in inputstr:
        if character == '1':
            outputlist.append(1)
        else:
            outputlist.append(0)
    return outputlist

true_samples = [str_to_list(x) for x in true_samples]
false_samples = [str_to_list(x) for x in false_samples]

# TODO use template for an abstract problem class
class StupidProblem():
    def __init__(self):
        self.true_samples = true_samples
        self.false_samples = false_samples
        self.input_arity = 4
        self.output_arity = 1

    def get_problem_instance(self):
        return randchoice(self.true_samples + self.false_samples)

    # MSE, sort of
    def error(self, problem, solution, logger):
        solution = solution[0]
        if solution is None:
            logger.log("instance_solution", f"None, {problem}")
            return 1, False
        if problem in self.true_samples:
            logger.log("instance_solution", f"True sample, prediction {solution}, problem {problem}")
            return min(1, (1-solution)**2), True
        else:
            logger.log("instance_solution", f"True sample, prediction {solution}, problem {problem}")
            return min(1, (solution)**2), True

    def get_reward(self, error):
        return 1/(1+abs(1-error))      
            