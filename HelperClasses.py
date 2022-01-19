import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import time




class Counter:
    def __init__(self) -> None:
        self.count = 0

    def counterval(self):
        self.count += 1
        return self.count


#seed = 100
#random.seed(seed)
def randchoice(alternative_list):
    randval = random.random()
    max = len(alternative_list)-1
    index = int(math.ceil(randval*max))
    return alternative_list[index]

def randchoice_scaled(alternative_list, value_scaling):
    # Normalize value_scaling values to sum to 1, then calculate valule intervals and pick random
    # value
    if len(alternative_list) != len(value_scaling):
        raise Exception(f"Alternative length and value scaling lengths are not equal, respectively {len(alternative_list)} and {len(value_scaling)}")
    value_scaling_sum = sum(value_scaling)
    values_scaled = [x/value_scaling_sum for x in value_scaling]
    randval = random.random()
    value_interval = []
    value_interval.append(values_scaled[0])
    for num in range(1, len(values_scaled)):
        value_interval.append(values_scaled[num] + value_interval[num-1])
    index = 0
    for value in value_interval:
        if randval <= value:
            return alternative_list[index]
        index += 1
    raise Exception("randchoice_scaled function is broken - no value found in scaled interval")

def drawProgram(active_nodes, output_nodes, input_nodes):
    input_node_types =  [f"{node.type}-{node.id}-{node.arity}" for node in input_nodes]
    output_node_types =  [f"{node.type}-{node.id}-{node.arity}" for node in output_nodes]
    node_types = [f"{node.type}-{node.id}-{node.arity}" for node in active_nodes + input_nodes]
    connection_pairs = []
    for node in active_nodes + output_nodes + input_nodes:
        for subscriber in node.subscribers:
            if subscriber in active_nodes + output_nodes + input_nodes:
                connection_pairs.append((f"{node.type}-{node.id}-{node.arity}", f"{subscriber.type}-{subscriber.id}-{subscriber.arity}"))
        for input_node in node.inputs:
            if input_node in active_nodes + output_nodes + input_nodes:
                connection_pairs.append((f"{input_node.type}-{input_node.id}-{input_node.arity}", f"{node.type}-{node.id}-{node.arity}"))



    g = nx.DiGraph()
    g.add_nodes_from(node_types)
    for pair in connection_pairs:
        g.add_edge(pair[0], pair[1])
    color_map = []
    for node in g.nodes:
        if node in input_node_types:
            color_map.append('green')
        elif node in output_node_types:
            color_map.append('red')
        else:
            color_map.append('blue')
    nx.draw(g, node_color = color_map, with_labels = False)
    labels = {node:node for node in node_types + output_node_types}
    nx.draw_networkx_labels(g, nx.spring_layout(g), labels, font_size = 16, font_color='b')
    print(g.nodes)
    print(g.edges)
    plt.draw()
    plt.show()
    #plt.savefig(str(time.time()) + ".png", format="PNG")

def randcheck(val):
    return random.random() <= val

def listmult(the_list, val):
    val = min(val, 1.0)
    return [x*val for x in the_list]


def copydict(input_dict):
    newdict = {}
    if type(input_dict) == dict:
        for key, item in input_dict.items():
            newdict[key] = copydict(item)
        return newdict
    else:
        return input_dict

