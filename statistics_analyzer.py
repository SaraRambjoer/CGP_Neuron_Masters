import yaml as yaml
import numpy as np 
import os.path as path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas


def average(a_list):
    return sum(a_list)/len(a_list)

statistics_folder = r"C:\Users\jonora\Documents\logfile_test"
statistics_files = [r"C:\Users\jonora\Documents\logfile_test\statistics.yml"]


# Assumes all stat files have the same amount of iterations and tracks all the required statisticks and same amount of genomes
yaml_stats = []

for statistics_file in statistics_files:
    with open(statistics_file, 'r') as f:
        yaml_stats.append(yaml.load(f, Loader=yaml.SafeLoader))

# TODO ANALYZE 'replacement_stats'

def plot_basic(data, ylabel, xlabel, figname):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

def plot_multiple(datas, labels, ylabel, xlabel, figname):
    for num in range(len(datas)):
        plt.plot(datas[num])
    plt.legend(labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())


# graphs for genome replacement stats
genome_takeover_probabilities = []
genome_takeover_counts = []
for it in range(len(yaml_stats[0]['iterations'])):
    genome_takeover_probabilities += [stat['iterations'][it]['genome_replacement_stats']['average_takeover_probability'] for stat in yaml_stats]
    genome_takeover_counts += [stat['iterations'][it]['genome_replacement_stats']['times_a_genome_took_population_slot_from_other_genome'] for stat in yaml_stats]


plot_basic(genome_takeover_probabilities, "Avg. chance of genome replacing another in pop.", "Iteration", "avg_genome_replacement.png")
plot_basic(genome_takeover_counts, "Genome takeover counts", "Iteration", "genome_takeover_count.png")

# CGP node type trends: 
cgp_node_type_data = {}
cgp_node_types = []


for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    for genome_list in genome_lists:
        for genome in genome_list:
            inner_cgp_node_types = genome['cgp_node_types']
            for key in inner_cgp_node_types.keys():
                if key not in cgp_node_types:
                    cgp_node_types.append(key)

cgp_node_type_data = {x:[] for x in cgp_node_types}

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    inner_cgp_node_type_data = {x:0 for x in cgp_node_types}
    for genome_list in genome_lists:
        for genome in genome_list:
            inner_cgp_node_types = genome['cgp_node_types']
            for key, val in inner_cgp_node_types.items():
                inner_cgp_node_type_data[key] += val
    for key, val in inner_cgp_node_type_data.items():
        cgp_node_type_data[key].append(val)

plot_multiple(
    [val for _, val in cgp_node_type_data.items()],
    [key for key, _ in cgp_node_type_data.items()],
    "count",
    "iteration",
    "cgp_node_type_trends.png"
)

# Fitness over time
max_fitness = []
min_fitness = []
avg_fitness = []
std_fitness = []
for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]

    fitnessess = [x['fitness'] for x in genome_list for genome_list in genome_lists]
    max_fitness.append(max(fitnessess))
    min_fitness.append(min(fitnessess))
    avg_fitness.append(average(fitnessess))
    std_fitness.append(np.std(fitnessess))

plot_multiple(
    [max_fitness, min_fitness, avg_fitness, std_fitness],
    ["max", "min", "average", "std."],
    "float",
    "iteration",
    "fitness.png"
)

# mutation chances
max_link_mutation_chances = []
min_link_mutation_chances = []
avg_link_mutation_chances = []
std_link_mutation_chances = []

max_node_mutation_chances = []
min_node_mutation_chances = []
avg_node_mutation_chances = []
std_node_mutation_chances = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]

    link_mutation_chances = [x['link_mutation_chance'] for x in genome_list for genome_list in genome_lists]
    node_mutation_chances = [x['node_mutation_chance'] for x in genome_list for genome_list in genome_lists]
    max_link_mutation_chances.append(max(link_mutation_chances))
    min_link_mutation_chances.append(min(link_mutation_chances))
    avg_link_mutation_chances.append(average(link_mutation_chances))
    std_link_mutation_chances.append(np.std(link_mutation_chances))
    max_node_mutation_chances.append(max(node_mutation_chances))
    min_node_mutation_chances.append(min(node_mutation_chances))
    avg_node_mutation_chances.append(average(node_mutation_chances))
    std_node_mutation_chances.append(np.std(node_mutation_chances))

plot_multiple(
    [max_link_mutation_chances, min_link_mutation_chances, avg_link_mutation_chances, std_link_mutation_chances],
    ["max", "min", "average", "std."],
    "float",
    "iteration",
    "link_mutation_chance.png"
)

plot_multiple(
    [max_node_mutation_chances, min_node_mutation_chances, avg_node_mutation_chances, std_node_mutation_chances],
    ["max", "min", "average", "std."],
    "float",
    "iteration",
    "node_mutation_chance.png"
)

# hox switch count

hox_switch_count_average = []
hox_switch_count_std = []
for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    hox_switch_counts = [x['performance_stats']['hox_switch_count'] for x in genome_list for genome_list in genome_lists]
    hox_switch_count_average.append(average(hox_switch_counts))
    hox_switch_count_std.append(np.std(hox_switch_counts))

plot_multiple(
    [hox_switch_count_average, hox_switch_count_std],
    ["avg", "std"],
    "float",
    "iteration",
    "hox_switch.png"
)

# connectivity
node_connectivity_avg = []
node_connectivity_std = []
input_node_connectivity_avg = []
input_node_connectivity_std = []
output_node_connectivity_std = []
output_node_connectivity_avg = []


for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    node_connectivities = [x['performance_stats']['node_connectivity'] for x in genome_list for genome_list in genome_lists]
    input_node_connectivities = [x['performance_stats']['input_connectivity'] for x in genome_list for genome_list in genome_lists]
    output_node_connectivities = [x['performance_stats']['output_connectivity'] for x in genome_list for genome_list in genome_lists]
    node_connectivity_avg.append(average(node_connectivities))
    node_connectivity_std.append(np.std(node_connectivities))
    input_node_connectivity_avg.append(average(input_node_connectivities))
    input_node_connectivity_std.append(np.std(input_node_connectivities))
    output_node_connectivity_avg.append(average(output_node_connectivities))
    output_node_connectivity_std.append(np.std(output_node_connectivities))

plot_multiple(
    [node_connectivity_avg, node_connectivity_std, input_node_connectivity_avg, input_node_connectivity_std, output_node_connectivity_std, output_node_connectivity_avg],
    ['node avg', 'node std', 'input avg', 'input std', 'output avg', 'output std'],
    'connectivity',
    'iteration',
    'connectivity.png'
)

# Unique input/output connections

unique_output_node_connections_avg = []
unique_input_node_connections_avg = []
unique_output_node_connections_std = []
unique_input_node_connections_std = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    unique_output_node_connections = [x['performance_stats']['nodes_connected_to_output_nodes'] for x in genome_list for genome_list in genome_lists]
    unique_input_node_connections = [x['performance_stats']['nodes_connected_to_input_nodes'] for x in genome_list for genome_list in genome_lists]
    unique_output_node_connections_avg.append(average(unique_output_node_connections))
    unique_input_node_connections_avg.append(average(unique_input_node_connections))
    unique_output_node_connections_std.append(np.std(unique_output_node_connections))
    unique_input_node_connections_std.append(np.std(unique_input_node_connections))

plot_multiple(
    [unique_output_node_connections_avg, unique_output_node_connections_std, unique_input_node_connections_avg, unique_input_node_connections_std],
    ['output avg', 'output std', 'input avg', 'input std'],
    'unique connected nodes to layer', 
    'iteration',
    'in_out_node_connections_unique.png'
)


module_size_avg = []
module_size_std = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    module_size_avg.append(average([x['module_size_average'] for x in genome_list for genome_list in genome_lists]))
    module_size_std.append(np.std([x['module_size_average'] for x in genome_list for genome_list in genome_lists]))

plot_multiple(
    [module_size_avg, module_size_std],
    ['module size avg', 'module size std'],
    'module size in nodes',
    'iteration',
    'module_size.png'
)

module_count_avg = []
module_count_std = []
recursive_module_count_avg = []
recursive_module_count_std = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    module_count_avg.append(average([x['module_count_non_recursive'] for x in genome_list for genome_list in genome_lists]))
    module_count_std.append(np.std([x['module_count_non_recursive'] for x in genome_list for genome_list in genome_lists]))
    recursive_module_count_avg.append(average([x['module_count_recursive'] for x in genome_list for genome_list in genome_lists]))
    recursive_module_count_std.append(np.std([x['module_count_recursive'] for x in genome_list for genome_list in genome_lists]))

plot_multiple(
    [module_count_avg, module_count_std, recursive_module_count_avg, recursive_module_count_std],
    ['module count average', 'module count std', 'recursive module count average', 'recursive module count std'],
    'Averaeg module count per genome',
    'iteration',
    'module_count.png'
)

max_module_depth_average = []
max_module_depth_std = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    max_module_depth_average.append(average([x['module_max_depth'] for x in genome_list for genome_list in genome_lists]))
    max_module_depth_std.append(np.std([x['module_max_depth'] for x in genome_list for genome_list in genome_lists]))

plot_multiple(
    [max_module_depth_average, max_module_depth_std],
    ['Average max module depth', 'Max module depth std'],
    'Iteration',
    'max_module_depth.png'
)

total_active_nodes_average = []
total_active_nodes_std = []

for it in range(len(yaml_stats[0]['iterations'])):
    genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
    total_active_nodes_average.append(average([x['total_active_nodes'] for x in genome_list for genome_list in genome_lists]))
    total_active_nodes_std.append(np.std([x['total_active_nodes'] for x in genome_list for genome_list in genome_lists]))


plot_multiple(
    [total_active_nodes_average, total_active_nodes_std],
    ['Total active nodes avg.', 'Total active nodes std.'],
    'Iteration',
    'total_active_nodes.png'
)



dataframe = pandas.DataFrame(
    data = [
        total_active_nodes_average,
        max_module_depth_average,
        module_count_avg,
        recursive_module_count_avg,
        module_size_avg,
        unique_output_node_connections_avg,
        unique_input_node_connections_avg,
        node_connectivity_avg,
        input_node_connectivity_avg,
        output_node_connectivity_avg,
        hox_switch_count_average,
        max_link_mutation_chances,
        min_link_mutation_chances,
        avg_link_mutation_chances,
        max_node_mutation_chances,
        min_node_mutation_chances,
        avg_node_mutation_chances,
        max_fitness,
        min_fitness,
        avg_fitness,
        std_fitness,
        genome_takeover_probabilities,
        genome_takeover_counts
    ] + cgp_node_type_data.values(),
    columns=[
        'total_active_nodes_average',
        'max_module_depth_average',
        'module_count_avg',
        'recursive_module_count_avg',
        'module_size_avg',
        'unique_output_node_connections_avg',
        'unique_input_node_connections_avg',
        'node_connectivity_avg',
        'input_node_connectivity_avg',
        'output_node_connectivity_avg',
        'hox_switch_count_average',
        'max_link_mutation_chances',
        'min_link_mutation_chances',
        'avg_link_mutation_chances',
        'max_node_mutation_chances',
        'min_node_mutation_chances',
        'avg_node_mutation_chances',
        'max_fitness',
        'min_fitness',
        'avg_fitness',
        'std_fitness',
        'genome_takeover_probabilities',
        'genome_takeover_counts',
    ] + cgp_node_type_data.keys()
)



correlation_mat = dataframe.corr()

sns.heatmap(correlation_mat, annot = True)

plt.savefig('correlation_matrix.png')
