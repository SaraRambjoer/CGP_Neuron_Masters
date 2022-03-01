from unittest import skip

filepath = r"C:\Users\jonora\Documents\CGP Neuron Masters\statistics.yml"

output = []
iterations_found = 0
with open(filepath, "r") as f:
    for line in f:
        if "iterations:" in line:
            if iterations_found < 2:
                output.append(line)
                iterations_found += 1
        else:
            output.append(line)

with open(filepath, "w") as f:
    f.writelines(output)

