# script that checks if all neurons connected to an input neuron is also connected to an output neuron. 
import json

filepath = r"C:\Users\jonod\Desktop\logfiles_15_april\config_complex_2\2\log_config_complex_2_1649355876.175315\graphlog_run.txt"
with open(filepath, "r") as f:
    textdata = f.read()

for entry in textdata.split("|"):
    entry = json.loads(entry)
    inputs = entry['inputs']
    outputs = entry['outputs']
    connections = entry['connections']
    for input_neuron in inputs: 
        relevant_connections = [x[1] for x in connections if x[0] == input_neuron]
        for neuron in relevant_connections:
            second_degree_connections = [x[1] for x in connections if x[0] == neuron]
            for output in outputs: 
                if output not in second_degree_connections:
                    print("False")
print("True")