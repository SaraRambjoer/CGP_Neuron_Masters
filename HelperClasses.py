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
