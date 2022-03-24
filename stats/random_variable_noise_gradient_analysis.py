import numpy as np
import random
import math
# A script for evaluating how much the gradient dissapears under the noise of fitness random variables


def estimate(population, mean, better_prob, better_val, variance):
    better_count = 0
    worse_count = 0
    neutral_count = 0
    for num in range(population):
        randval = random.random()
        if randval < better_prob:
            better_count += 1
        elif randval < (better_prob + (1.0-better_prob)/2):
            neutral_count += 1
        else:
            worse_count += 1

    better_samples = [1003]
    neutral_samples = [1002] # just to have defaults for max
    worse_samples = [1001]
    for num in range(better_count):
        better_samples.append(
            max(0.0, np.random.normal(mean-better_val, variance))
        )
    for num in range(neutral_count):
        neutral_samples.append(
            max(0.0, np.random.normal(mean, variance))
        )
    for num in range(worse_count):
        worse_samples.append(
            max(0.0, np.random.normal(mean+better_val, variance))
        )
    better_min = min(better_samples)
    neutral_min = min(neutral_samples)
    worse_min = min(worse_samples)
    if better_count != 0 and (neutral_count == 0 or better_min < neutral_min) and (worse_count == 0 and better_min < worse_min):
        if better_min >= mean:
            return 0
        return 1
    elif neutral_count != 0 and (worse_count == 0 or neutral_min < worse_min):
        return 0
    else:
        if worse_min >= mean:
            return 0
        return -1

def format_spec(val):
    if val < 0.01:
        return "{:.2E}".format(val)
    else:
        return "{:.2f}".format(val)

population = 4
mean = 0.09
better_values = [mean*0.0001, mean*0.001, mean*0.01, mean*0.1, mean*0.25]
variances = [0.003, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
better_prob = [0.001, 0.01, 0.05, 0.1, 0.2]
for bet_count in better_prob:
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\begin{adjustwidth}{-1.0in}{}")
    print(r"\begin{tabularx}{\linewidth}{|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|}")
    print(r"\hline")
    print("Fitness improvement, std", end=" & ")
    for bet_val in better_values:
        print(format_spec(bet_val), end=" & ")
    print(r"\\\hline")
    for var in variances:
        print(format_spec(var), end=" & ")
        for bet_val in better_values:
            better_res_count = 0
            neutral_res_count = 0
            worse_res_count = 0
            for num in range(25000):
                res = estimate(population, mean, bet_count, bet_val, var)
                if res == 1:
                    better_res_count += 1
                elif res == 0: 
                    neutral_res_count += 1
                else:
                    worse_res_count += 1
            if bet_val != better_values[-1]:
                print(format_spec(better_res_count/25000) + " / " + format_spec(neutral_res_count/25000) + " / " + format_spec(worse_res_count/25000), end=" & ")
            else:
                print(format_spec(better_res_count/25000) + " / " + format_spec(neutral_res_count/25000) + " / " + format_spec(worse_res_count/25000), end=" ")
        print(r"\\\hline")
    print(r"\end{tabularx}")
    print(r"\end{adjustwidth}")
    print(r"\caption{Percentage of respectively actually good steps/neutral steps/worse steps in state space for different variances and step sizes if " + str(bet_count) + " of the child genotypes are better than the parent. For reference, the actual percentage of the time a deterministic gradient descent algorithm could descend the gradient is the amount of children times the chance of a child being better ("+ str(4*bet_count)+ ")}")
    print(r"\label{tab:gradient_dissaperance_better_count" + str(bet_count) + "}")
    print(r"\end{table}")
