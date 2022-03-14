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
from scipy.stats import entropy
import os


runs_per_iteration = 125

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

def plot_basic(data, ylabel, xlabel, figname, statistics_folder):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

def plot_multiple(datas, labels, ylabel, xlabel, figname, statistics_folder):
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

def plot_average_std(datas, labels, ylabel, xlabel, figname, statistics_folder):
    # assumes data order: Average, std
    plt.plot(datas[0])
    plt.fill_between(range(len(datas[0])), listmin(datas[0], datas[1]), listplus(datas[0], datas[1]), alpha=0.5)
    plt.legend(labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path.join(statistics_folder, figname))
    plt.close(plt.gcf())

def plot_average_std_double(datas, labels, ylabel, xlabel, figname, statistics_folder):
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

def plot_average_std_min_max(datas, labels, ylabel, xlabel, figname, statistics_folder):
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

def runme(statistics_folder, statistics_files):
    print(statistics_files)
    statistics_files = statistics_files.split("SPLIT")
    print(statistics_files)

    # Assumes all stat files have the same amount of iterations and tracks all the required statisticks and same amount of genomes
    yaml_stats = []

    for statistics_file in statistics_files:
        with open(statistics_file, 'r') as f:
            yaml_stats.append(yaml.load(f, Loader=yaml.SafeLoader))
    # graphs for genome replacement stats
    genome_takeover_counts = []
    for it in range(len(yaml_stats[0]['iterations'])):
        genome_takeover_counts.append(average([stat['iterations'][it]['genome_replacement_stats']['times_a_genome_took_population_slot_from_other_genome'] for stat in yaml_stats]))


    plot_basic(genome_takeover_counts, "Genome takeover counts", "Iteration", "genome_takeover_count.png", statistics_folder)

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
        "cgp_node_type_trends.png", statistics_folder
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
        "fitness.png", statistics_folder
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
        "link_mutation_chance.png", statistics_folder
    )

    plot_average_std_min_max(
        [avg_node_mutation_chances, std_node_mutation_chances, min_node_mutation_chances, max_node_mutation_chances],
        ["average", "std.", "min", "max"],
        "node mutation chance",
        "iteration",
        "node_mutation_chance.png", statistics_folder
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
        "hox_switch.png", statistics_folder
    )

    # connectivity
    neuron_connectivity_avg = []
    neuron_connectivity_std = []
    input_neuron_connectivity_avg = []
    input_neuron_connectivity_std = []
    output_neuron_connectivity_std = []
    output_neuron_connectivity_avg = []


    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]

        node_connectivities = []
        input_node_connectivities = []
        output_node_connectivities = []
        for genome_list in genome_lists:
            for genome in genome_list:
                node_connectivities.append(genome['performance_stats']['neuron_connectivity'])
                input_node_connectivities.append(genome['performance_stats']['input_connectivity'])
                output_node_connectivities.append(genome['performance_stats']['output_connectivity'])
        neuron_connectivity_avg.append(average(node_connectivities))
        neuron_connectivity_std.append(np.std(node_connectivities))
        input_neuron_connectivity_avg.append(average(input_node_connectivities))
        input_neuron_connectivity_std.append(np.std(input_node_connectivities))
        output_neuron_connectivity_avg.append(average(output_node_connectivities))
        output_neuron_connectivity_std.append(np.std(output_node_connectivities))

    plot_average_std(
        [neuron_connectivity_avg, neuron_connectivity_std],
        ["node avg", "node_std"],
        "connectivity",
        "iteration",
        "connectivity.png", statistics_folder
    )

    plot_average_std(
        [input_neuron_connectivity_avg, input_neuron_connectivity_std],
        ["input node avg", "input node std"],
        "connectivity",
        "iteration",
        "connectivity_input.png", statistics_folder
    )

    plot_average_std(
        [output_neuron_connectivity_avg, output_neuron_connectivity_std],
        ["output node avg", "output node std"],
        "connectivity",
        "iteration",
        "connectivity_output.png", statistics_folder
    )

    # Unique input/output connections

    unique_output_neuron_connections_avg = []
    unique_input_neuron_connections_avg = []
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

        unique_output_neuron_connections_avg.append(average(unique_output_node_connections))
        unique_input_neuron_connections_avg.append(average(unique_input_node_connections))
        unique_output_node_connections_std.append(np.std(unique_output_node_connections))
        unique_input_node_connections_std.append(np.std(unique_input_node_connections))

    plot_average_std(
        [unique_output_neuron_connections_avg, unique_output_node_connections_std],
        ["output avg", "output std"],
        "unique connected neurons to layer",
        "iteration",
        "out_connections.png", statistics_folder
    )

    plot_average_std(
        [unique_input_neuron_connections_avg, unique_input_node_connections_std],
        ["input avg", "input std"],
        "unique connected neurons to layer",
        "iteration",
        "in_connections.png", statistics_folder
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
        'module_size.png', statistics_folder
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
        'module_count.png', statistics_folder
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
        'max_module_depth.png', statistics_folder
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
        'total_active_nodes.png', statistics_folder
    )

    neuron_counts_avg = []
    neuron_counts_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        genome_lists = [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]
        neuron_counts = []
        for genome_list in genome_lists:
            for genome in genome_list:
                neuron_counts.append(genome['performance_stats']['neuron_count']) 

        neuron_counts_avg.append(average(neuron_counts))
        neuron_counts_std.append(np.std(neuron_counts))

    plot_average_std(
        [neuron_counts_avg, neuron_counts_std],
        ["Average neuron count", "Neuron count std."],
        "Neuron phenotype count",
        "Iteration",
        "neuron_phenotype_count.png", statistics_folder
    )

    population_entropy_avg = []
    for it in range(len(yaml_stats[0]['iterations'])):
        inner_avgs = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            id_strings = []
            for genome in genome_dat['genomes_data']['genome_list']:
                id_strings.append(genome['id'])
            encountered = {}
            for id in id_strings:
                if id not in encountered:
                    encountered[id] = 1
                else:
                    encountered[id] += 1
            inner_avgs.append(entropy(list(encountered.values())))
        population_entropy_avg.append(average(inner_avgs))
    
    plot_basic(population_entropy_avg,
        "Average shannon entropy of IDs",
        "Iterations",
        "population_shannon_entropy.png",
        statistics_folder
    )

        
    
    

    better_swap_avg = []
    better_swap_std = []
    neutral_swap_avg = []
    neutral_swap_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        better_swap = []
        neutral_swap = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            better_swap.append(genome_dat['replacement_stats']['better_population_swaps_percentage']) 
            neutral_swap.append(genome_dat['replacement_stats']['neutral_population_swaps_percentage']) 
        
        

        better_swap_avg.append(average(better_swap))
        neutral_swap_avg.append(average(neutral_swap))
        better_swap_std.append(np.std(better_swap))
        neutral_swap_std.append(np.std(neutral_swap_std))

    any_swap_avg = listplus(better_swap_avg, neutral_swap_avg)

    plot_average_std(
        [better_swap_avg, better_swap_std],
        ["Better swap %", "std"],
        "%",
        "Iteration",
        "better_swap.png", statistics_folder
    )

    plot_average_std(
        [neutral_swap_avg, neutral_swap_std],
        ["Neutral swap %", "std"],
        "%",
        "Iteration",
        "neutral_swap.png", statistics_folder
    )

    plot_basic(any_swap_avg, "Any swap %", "Iteration", "any_swap.png", statistics_folder)

    historic_best_swap_avg = []

    for it in range(len(yaml_stats[0]['iterations'])):
        historic_swap = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            historic_swap.append(genome_dat['replacement_stats']['historic_best_swaps'])

        historic_best_swap_avg.append(average(historic_swap))

    plot_basic(historic_best_swap_avg, "Historic list swaps", "Iteration", "historic_best_swaps.png", statistics_folder)


    better_child_percentage_avg = []
    better_child_percentage_std = []
    neutral_child_percentage_avg = []
    neutral_child_percentage_std = []

    for it in range(len(yaml_stats[0]['iterations'])):
        better_child_percentage = []
        neutral_child_percentage = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            better_child_percentage.append(genome_dat['replacement_stats']['better_child_percentage']) 
            neutral_child_percentage.append(genome_dat['replacement_stats']['neutral_child_percentage']) 
        
        

        better_child_percentage_avg.append(average(better_child_percentage))
        neutral_child_percentage_avg.append(average(neutral_child_percentage))
        better_child_percentage_std.append(np.std(better_child_percentage))
        neutral_child_percentage_std.append(np.std(neutral_child_percentage_std))

    any_change_avg = listplus(better_child_percentage_avg, neutral_child_percentage_avg)

    plot_average_std(
        [better_child_percentage_avg, better_child_percentage_std],
        ["Better child %", "std"],
        "%",
        "Iteration",
        "better_child.png", statistics_folder
    )

    plot_average_std(
        [neutral_child_percentage_avg, neutral_child_percentage_std],
        ["Neutral child %", "std"],
        "%",
        "Iteration",
        "neutral_child.png", statistics_folder
    )
    plot_basic(any_change_avg, "Any change %", "Iteration", "any_change.png", statistics_folder)

    dendrite_internal_state_use_count_average = []
    constant_number_use_avg = []
    neuron_engine_dim_use_avg = []
    signal_dim_use_avg = []
    neuron_internal_state_avg = []

    for it in range(len(yaml_stats[0]['iterations'])):
        dendrite_use_avg = []
        constant_avg = []
        neuron_dim_avg = []
        signal_dim_avg = []
        neuron_int_avg = []
        for genome_dat in [x['iterations'][it]['genomes_data'] for x in yaml_stats]:
            dendrite_use_avg.append(genome_dat['dendrite_internal_state_use_count'])
            constant_avg.append(genome_dat['constant_number_use_count'])
            neuron_dim_avg.append(genome_dat['neuron_engine_dimensionality_use_count'])
            signal_dim_avg.append(genome_dat['signal_dimensionality_use_count'])
            neuron_int_avg.append(genome_dat['neuron_internal_state_use_count'])
        
        dendrite_internal_state_use_count_average.append(average(dendrite_use_avg))
        constant_number_use_avg.append(average(constant_avg))
        neuron_engine_dim_use_avg.append(average(neuron_dim_avg))
        signal_dim_use_avg.append(average(signal_dim_avg))
        neuron_internal_state_avg.append(average(neuron_int_avg))
    
    plot_basic(dendrite_internal_state_use_count_average, "Avg. CGP connection count to dendrite state input", "Iteration", "cgp_use_dendrite_state.png", statistics_folder)
    plot_basic(constant_number_use_avg, "Avg. CGP connection count to constant number input", "Iteration", "cgp_use_constant.png", statistics_folder)
    plot_basic(neuron_engine_dim_use_avg, "Avg. CGP connection count to neuron engine dimension input", "Iteration", "cgp_use_neuron_engine.png", statistics_folder)
    plot_basic(signal_dim_use_avg, "Avg. CGP connection count to signal dimension input", "Iteration", "cgp_use_signal_dim.png", statistics_folder)
    plot_basic(neuron_internal_state_avg, "Avg. CGP connection count to neuron state input", "Iteration", "cgp_use_neuron_state.png", statistics_folder)

    corrected_fitness_average = []
    for it in range(len(yaml_stats[0]['iterations'])):
        corr_fitness = []
        for genome_dat in [x['iterations'][it]['genomes_data'] for x in yaml_stats]:
            for genome_list in genome_dat['genome_list']:
                fitness = genome_list['fitness']
                establishment_time = genome_list['performance_stats']['no outputs']
                corrected = fitness*(runs_per_iteration-establishment_time)/runs_per_iteration
                corr_fitness.append(corrected)
        corrected_fitness_average.append(average(corr_fitness))
    
    plot_basic(corrected_fitness_average, "Corrected fitness by establishment time (for pole balancing)", "Iteration", "corrected_fitness.png", statistics_folder)


    _fldr = os.path.join(statistics_folder, "neuron_engine_actions")
    if not os.path.exists(_fldr):
        os.mkdir(_fldr)
    
    for name in [
        'axon_recieve_signal_dendrite',
        'axon_recieve_signal_neuron',
        'axon_signal_dendrite',
        'dendrite_accept_connection',
        'dendrite_action_controller',
        'dendrite_axon_death_connection_signal',
        'dendrite_axon_death_neuron_signal',
        'dendrite_break_connection',
        'dendrite_die',
        'dendrite_recieve_reward',
        'dendrite_recieve_signal_axon',
        'dendrite_recieve_signal_dendrite',
        'dendrite_recieve_signal_neuron',
        'dendrite_seek_connection',
        'dendrite_signal_axon',
        'dendrite_signal_dendrite',
        'dendrite_signal_neuron',
        'neuron_action_controller',
        'neuron_axon_birth',
        'neuron_dendrite_birth',
        'neuron_die',
        'neuron_hox_variant_selection',
        'neuron_move',
        'neuron_neuron_birth',
        'neuron_recieve_axon_signal',
        'neuron_recieve_reward',
        'neuron_signal_axon',
        'neuron_signal_dendrite',
        'skip_post_death'
    ]:
        avg = []
        for it in range(len(yaml_stats[0]['iterations'])):
            times = []
            for genome_dat in [x['iterations'][it]['genomes_data']['genome_list'] for x in yaml_stats]:
                for genome_entry in genome_dat:
                    times.append(genome_entry['actions_stats'][name])
            avg.append(average(times))

        plot_basic(avg, name, "Iteration", name+".png", _fldr)





    eval_time_averages = []

    for it in range(len(yaml_stats[0]['iterations'])):
        eval_times = []
        for genome_dat in [x['iterations'][it] for x in yaml_stats]:
            eval_times.append(genome_dat['time']['eval'])
        eval_time_averages.append(average(eval_times))
    
    plot_basic(eval_time_averages, "Eval time average", "Iteration", "eval_times.png", statistics_folder)


    x = [
            total_active_nodes_average,
            max_module_depth_average,
            module_count_avg,
            recursive_module_count_avg,
            module_size_avg,
            unique_output_neuron_connections_avg,
            unique_input_neuron_connections_avg,
            neuron_connectivity_avg,
            input_neuron_connectivity_avg,
            output_neuron_connectivity_avg,
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
            eval_time_averages,
            neuron_counts_avg,
            population_entropy_avg,
            better_swap_avg,
            neutral_swap_avg,
            any_swap_avg,
            better_child_percentage_avg,
            neutral_child_percentage_avg,
            any_change_avg,
            dendrite_internal_state_use_count_average,
            constant_number_use_avg,
            neuron_engine_dim_use_avg,
            signal_dim_use_avg,
            neuron_internal_state_avg
        ] + [[y for y in x] for x in cgp_node_type_data.values()]

    y = [
            'total_active_nodes_average',
            'max_module_depth_average',
            'module_count_avg',
            'recursive_module_count_avg',
            'module_size_avg',
            'unique_output_neuron_connections_avg',
            'unique_input_neuron_connections_avg',
            'neuron_connectivity_avg',
            'input_neuron_connectivity_avg',
            'output_neuron_connectivity_avg',
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
            'eval_time_averages',
            'neuron_counts_avg',
            "population_id_entropy_avg",
            'better_swap_avg',
            'neutral_swap_avg',
            'any_swap_avg',
            'better_child_percentage',
            'neutral_child_percentage',
            'any_change_child_percentage',
            'dendrite_internal_state_use_count_average',
            'constant_number_use_avg',
            'neuron_engine_dim_use_avg',
            'signal_dim_use_avg',
            'neuron_internal_state_avg'

        ] + [x for x in cgp_node_type_data.keys()]
    
    dataframe = pandas.DataFrame(
        data = {x[0]:x[1] for x in zip(y, x)},
        columns= y
    )

    plot_corr(dataframe)

    plt.savefig(path.join(statistics_folder, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(plt.gcf())
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.close(plt.gcf()) 


if __name__ == '__main__':
    statistics_folder = sys.argv[1]
    statistics_files = sys.argv[2]
    runme(statistics_folder, statistics_files)
