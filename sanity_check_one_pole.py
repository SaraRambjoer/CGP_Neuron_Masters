from os import environ
import one_pole_problem
from HelperClasses import randchoice
from matplotlib import pyplot as plt
import numpy as np


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

fitnessess_flat = [np.average(x_inner) for x_inner in runs]
print("Standard deviation of fitness: ", np.std(fitnessess_flat), "Average: ", np.average(fitnessess_flat))




avg = 0.10167700000000003
std = 0.0037520222547314385

fitnessess_flat = np.asarray(fitnessess_flat)
better_vals = [0.01, 0.02, 0.05, 0.08, 0.102]
for val in better_vals:
    derived_val = avg-val
    prob_singular = 1.0-len(fitnessess_flat[fitnessess_flat <= derived_val])/len(fitnessess_flat)
    prob = 1.0-prob_singular*prob_singular*prob_singular*prob_singular
    print(prob)

gauss_samples_better = []
for num in range(runs_total):
    samples = np.random.normal(avg, std, 4)
    gauss_samples_better.append(
        [
            len(samples[samples <= avg - better_vals[0]])/4,
            len(samples[samples <= avg - better_vals[1]])/4,
            len(samples[samples <= avg - better_vals[2]])/4,
            len(samples[samples <= avg - better_vals[3]])/4
        ]
    )

print(np.average(gauss_samples_better, 0))


