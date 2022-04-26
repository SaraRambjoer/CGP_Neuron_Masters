import gym
import random
from HelperClasses import randchoice
class IrisProblem:
    """Implements the IRIS flower classification problem"""
    def __init__(self, data):
        self.data = [x for x in data]
        random.shuffle(self.data)
        self.index = 0

        self.input_arity = 4
        self.output_arity = 3
        self.correct_solution = 0

    def get_problem_instance(self):
        to_return = self.data[self.index][0:4]
        self.correct_solution = self.data[self.index][4]
        self.index += 1
        if self.index == len(self.data):
            self.index = 0
            random.shuffle(self.data)
        return to_return

    # Target is to minimize this
    def error(self, problem, solution, logger, debug = False):
        # MSE effectively but taking it into account that some may be None
        output_on_correct = solution[self.correct_solution]
        if output_on_correct is None:
            return 1, False
        outputs_sum = 0
        for num in range(0, 3):
            if solution[num] is not None:
                outputs_sum += solution[num]
        if outputs_sum > 0:
            normalized_output = output_on_correct/outputs_sum
        else:
            normalized_output = output_on_correct
        return (1-normalized_output)*(1-normalized_output), True

    def get_reward(self, error):
        return 1 - error



