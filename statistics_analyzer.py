import yaml as yaml
import numpy as np 
import os.path as path
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas
import sys

def average(a_list):
    return sum(a_list)/len(a_list)


def plot_corr(df,size=40):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """
    f = plt.figure(figsize=(size, size))
    corr = df.corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=6)



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

def listplus(list1, list2):
    return [list1[x]+list2[x] for x in range(len(list1))]

def listmin(list1, list2):
    return [list1[x]-list2[x] for x in range(len(list1))]

def plot_average_std(datas, labels, ylabel, xlabel, figname):
    # assumes data order: Average, std
    plt.plot(datas[0])
    plt.fill_between(range(len(datas[0])), listmin(datas[0], datas[1]), listplus(datas[0], datas[1]), alpha=0.5)
    plt.legend(labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

def plot_average_std_double(datas, labels, ylabel, xlabel, figname):
    # assumes data order: Average, std
    plt.plot(datas[0])
    plt.fill_between(range(len(datas[0])), listmin(datas[0], datas[1]), listplus(datas[0], datas[1]), alpha=0.5)
    plt.plot(datas[2])
    plt.fill_between(range(len(datas[2])), listmin(datas[2], datas[3]), listplus(datas[2], datas[3]), alpha=0.5)
    plt.legend(labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

def plot_average_std_min_max(datas, labels, ylabel, xlabel, figname):
    # assumes data order: Average, std, min, max
    plt.plot(datas[0])
    plt.fill_between(range(len(datas[0])), listmin(datas[0], datas[1]), listplus(datas[0], datas[1]), alpha=0.5)
    plt.plot(datas[2])
    plt.plot(datas[3])
    plt.legend(labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

if __name__ == '__main__':
    statistics_folder = sys.argv[1]
    statistics_files = sys.argv[2]
    statistics_files = statistics_files.split("|")


    # Assumes all stat files have the same amount of iterations and tracks all the required statisticks and same amount of genomes
    yaml_stats = []

    for statistics_file in statistics_files:
        with open(statistics_file, 'r') as f:
            yaml_stats.append(yaml.load(f, Loader=yaml.SafeLoader))
    # graphs for genome replacement stats
    genome_takeover_counts = []
    for it in range(len(yaml_stats[0]['iterations'])):
        genome_takeover_counts += [stat['iterations'][it]['genome_replacement_stats']['times_a_genome_took_population_slot_from_other_genome'] for stat in yaml_stats]


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
        fitnessess = []
        fitnessess_std = []
        for genome_list in genome_lists:
            for genome in genome_list:
                fitnessess.append(genome['fitness'])
                fitnessess_std.append(genome['fitness_std'])
        max_fitness.append(max(fitnessess))
        min_fitness.append(min(fitnessess))
        avg_fitness.append(average(fitnessess))
        std_fitness.append(average(fitnessess_std))

    plot_average_std_min_max(
        [avg_fitness, std_fitness, min_fitness, max_fitness],
        ["average", "Avg. std. for evals over genomes", "min", "max"],
        "fitness",
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
        link_mutation_chances = []
        node_mutation_chances = []
        for genome_list in genome_lists:
            for genome in genome_list:
                link_mutation_chances.append(genome['link_mutation_chance'])
                node_mutation_chances.append(genome['node_mutation_chance'])
        max_link_mutation_chances.append(max(link_mutation_chances))
        min_link_mutation_chances.append(min(link_mutation_chances))
        avg_link_mutation_chances.append(average(link_mutation_chances))
        std_link_mutation_chances.append(np.std(link_mutation_chances))
        max_node_mutation_chances.append(max(node_mutation_chances))
        min_node_mutation_chances.append(min(node_mutation_chances))
        avg_node_mutation_chances.append(average(node_mutation_chances))
        std_node_mutation_chances.append(np.std(node_mutation_chances))

    plot_average_std_min_max(
        [avg_link_mutation_chances, std_link_mutation_chances, min_link_mutation_chances, max_link_mutation_chances],
        ["average", "std.", "min", "max"],
        "link mutation chance",
        "iteration",
        "link_mutation_chance.png"
    )

    plot_average_std_min_max(
        [avg_node_mutation_chances, std_node_mutation_chances, min_node_mutation_chances, max_node_mutation_chances],
        ["average", "std.", "min", "max"],
        "node mutation chance",
        "iteration",
        "node_mutation_chance.png"
    )

    # hox switch count

    hox_switch_count_average = []
    hox_switch_count_std = []
    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        hox_switch_counts = []
        for genome_list in genome_lists:
            for genome in genome_list:
                hox_switch_counts.append(genome['performance_stats']['hox_switch_count'])
        hox_switch_count_average.append(average(hox_switch_counts))
        hox_switch_count_std.append(np.std(hox_switch_counts))

    plot_average_std(
        [hox_switch_count_average, hox_switch_count_std],
        ["avg", "std"],
        "hox switches per genome evaluation",
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

        node_connectivities = []
        input_node_connectivities = []
        output_node_connectivities = []
        for genome_list in genome_lists:
            for genome in genome_list:
                node_connectivities.append(genome['performance_stats']['node_connectivity'])
                input_node_connectivities.append(genome['performance_stats']['input_connectivity'])
                output_node_connectivities.append(genome['performance_stats']['output_connectivity'])
        node_connectivity_avg.append(average(node_connectivities))
        node_connectivity_std.append(np.std(node_connectivities))
        input_node_connectivity_avg.append(average(input_node_connectivities))
        input_node_connectivity_std.append(np.std(input_node_connectivities))
        output_node_connectivity_avg.append(average(output_node_connectivities))
        output_node_connectivity_std.append(np.std(output_node_connectivities))

    plot_average_std(
        [node_connectivity_avg, node_connectivity_std],
        ["node avg", "node_std"],
        "connectivity",
        "iteration",
        "connectivity.png"
    )

    plot_average_std(
        [input_node_connectivity_avg, input_node_connectivity_std],
        ["input node avg", "input node std"],
        "connectivity",
        "iteration",
        "connectivity_input.png"
    )

    plot_average_std(
        [output_node_connectivity_avg, output_node_connectivity_std],
        ["output node avg", "output node std"],
        "connectivity",
        "iteration",
        "connectivity_output.png"
    )

    # Unique input/output connections

    unique_output_node_connections_avg = []
    unique_input_node_connections_avg = []
    unique_output_node_connections_std = []
    unique_input_node_connections_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        unique_output_node_connections = []
        unique_input_node_connections = []
        for genome_list in genome_lists:
            for genome in genome_list:
                unique_output_node_connections.append(genome['performance_stats']['nodes_connected_to_output_nodes'])
                unique_input_node_connections.append(genome['performance_stats']['nodes_connected_to_input_nodes'])

        unique_output_node_connections_avg.append(average(unique_output_node_connections))
        unique_input_node_connections_avg.append(average(unique_input_node_connections))
        unique_output_node_connections_std.append(np.std(unique_output_node_connections))
        unique_input_node_connections_std.append(np.std(unique_input_node_connections))

    plot_average_std(
        [unique_output_node_connections_avg, unique_output_node_connections_std],
        ["output avg", "output std"],
        "unique connected nodes to layer",
        "iteration",
        "out_connections.png"
    )

    plot_average_std(
        [unique_input_node_connections_avg, unique_input_node_connections_std],
        ["input avg", "input std"],
        "unique connected nodes to layer",
        "iteration",
        "in_connections.png"
    )



    module_size_avg = []
    module_size_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]

        module_sizes = []
        for genome_list in genome_lists:
            for genome in genome_list:
                module_sizes.append(genome['module_size_average'])


        module_size_avg.append(average(module_sizes))
        module_size_std.append(np.std(module_sizes))

    plot_average_std(
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

        module_count = []
        recursive_module_count = []
        for genome_list in genome_lists:
            for genome in genome_list:
                module_count.append(genome['module_count_non_recursive'])
                recursive_module_count.append(genome['module_count_recursive'])

        module_count_avg.append(average(module_count))
        module_count_std.append(np.std(module_count))
        recursive_module_count_avg.append(average(recursive_module_count))
        recursive_module_count_std.append(np.std(recursive_module_count))




    plot_average_std_double(
        [module_count_avg, module_count_std, recursive_module_count_avg, recursive_module_count_std],
        ['module count average', 'module count std', 'recursive module count average', 'recursive module count std'],
        'Average module count per genome',
        'iteration',
        'module_count.png'
    )

    max_module_depth_average = []
    max_module_depth_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        max_module_depth = []
        for genome_list in genome_lists:
            for genome in genome_list:
                max_module_depth.append(genome['module_max_depth']) 
        max_module_depth_average.append(average(max_module_depth))
        max_module_depth_std.append(np.std(max_module_depth))

    plot_multiple(
        [max_module_depth_average, max_module_depth_std],
        ['Average max module depth', 'Max module depth std'],
        'Max module depth',
        'Iteration',
        'max_module_depth.png'
    )

    total_active_nodes_average = []
    total_active_nodes_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        total_active_nodes = []
        for genome_list in genome_lists:
            for genome in genome_list:
                total_active_nodes.append(genome['total_active_nodes']) 

        total_active_nodes_average.append(average(total_active_nodes))
        total_active_nodes_std.append(np.std(total_active_nodes))


    plot_average_std(
        [total_active_nodes_average, total_active_nodes_std],
        ['Total active nodes avg.', 'Total active nodes std.'],
        'Total active nodes',
        'Iteration',
        'total_active_nodes.png'
    )

    neuron_counts_avg = []
    neuron_counts_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        neuron_counts = []
        for genome_list in genome_lists:
            for genome in genome_list:
                neuron_counts.append(genome['performance_stats']['node_count']) 

        neuron_counts_avg.append(average(neuron_counts))
        neuron_counts_std.append(np.std(neuron_counts))

    plot_average_std(
        [neuron_counts_avg, neuron_counts_std],
        ["Average neuron count", "Neuron count std."],
        "Neuron phenotype count",
        "Iteration",
        "neuron_phenotype_count.png"
    )

    better_change_avg = []
    better_change_std = []
    neutral_change_avg = []
    neutral_change_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        better_change = []
        neutral_change = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            better_change.append(genome_dat['replacement_stats']['better_changes_percentage']) 
            neutral_change.append(genome_dat['replacement_stats']['neutral_changes_percentage']) 
        
        

        better_change_avg.append(average(better_change))
        neutral_change_avg.append(average(neutral_change))
        better_change_std.append(np.std(better_change))
        neutral_change_std.append(np.std(neutral_change_std))

    any_change_avg = listplus(better_change_avg, neutral_change_avg)

    plot_average_std(
        [better_change_avg, better_change_std],
        ["Better change %", "std"],
        "%",
        "Iteration",
        "better_change.png"
    )

    plot_average_std(
        [neutral_change_avg, neutral_change_std],
        ["Neutral change %", "std"],
        "%",
        "Iteration",
        "neutral_change.png"
    )

    plot_basic(any_change_avg, "Any change %", "Iteration", "any_change.png")


    x = [
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
            genome_takeover_counts,
            better_change_avg,
            neutral_change_avg
        ] + [[y for y in x] for x in cgp_node_type_data.values()]

    y = [
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
            'genome_takeover_counts',
            'better_change_avg',
            'neutral_change_avg'
        ] + [x for x in cgp_node_type_data.keys()]

    dataframe = pandas.DataFrame(
        data = {x[0]:x[1] for x in zip(y, x)},
        columns= y
    )

    plot_corr(dataframe)

    plt.savefig(path.join(statistics_folder, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
