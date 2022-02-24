from unittest import skip

filepath = "D:\\jonod\\masters\\CGP_Neuron_Masters\\logfiles\\log1645528312.2566803\\statistics.yml"

output = []
first_iterations_found = True
with open(filepath, "r") as f:
    for line in f:
        if "iterations:" in line:
            if first_iterations_found:
                output.append(line)
                first_iterations_found = False
        else:
            output.append(line)

with open(filepath, "w") as f:
    f.writelines(output)

