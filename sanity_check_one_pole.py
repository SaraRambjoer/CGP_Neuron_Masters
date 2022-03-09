from os import environ
import one_pole_problem
from HelperClasses import randchoice
from matplotlib import pyplot as plt


runs = []
environment_timesteps_per_run = 100
runs_total = 10000

reset_count = 0
for runnum in range(runs_total):
    fitnessess = []
    problem = one_pole_problem.PoleBalancingProblem()
    for num in range(environment_timesteps_per_run):
        action = [randchoice([0, 1])]
        return_package = problem.error(problem, action, None, True)
        if len(return_package) == 2:
            fitnessess.append(return_package[0])
        else:
            fitnessess.append(return_package[0])
            reset_count += 1
    runs.append(fitnessess)

print("Resets per run: ", reset_count/runnum)

val = 0
for num in range(100):
    for fitness_list in runs:
        val += fitness_list[num]
    
print("Expected random fitness: ", val/(runs_total*environment_timesteps_per_run))

# expect 0.101643