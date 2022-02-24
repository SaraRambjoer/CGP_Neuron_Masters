import math
import random
from HelperClasses import Counter, randchoice, randcheck, randchoice_scaled
from numpy.core.fromnumeric import sort, var
from numpy.lib.function_base import average
from numpy.ma import count
import numpy


class NodeAbstract():
    """Abstract superclass for nodes"""
    def __init__(self) -> None:
        self.subscribers = []
        self.output = None
        self.CPGNodeTypes = [
            "ADDI",
            "SUBI",
            "SINU",
            "GAUS"
        ]

        self.NodeTypeArity = {
            "ADDI":2,
            "SUBI":2,
            "GAUS":2,
            "SINU":1,
        }
        self.oneary = [x[0] for x in self.NodeTypeArity.items() if x[1] == 1]
        self.twoary = [x[0] for x in self.NodeTypeArity.items() if x[1] == 2]
        self.type_func_map = {
            "ADDI": self._addi,
            "SUBI": self._subi,
            "SINU": self._sinu,
            "GAUS": self._gaus
        }  

    def _addi(self, x0, x1):
        return x0+x1
    def _subi(self, x0, x1):
        return x0-x1
    def _sinu(self, x0):
        return numpy.sin(x0)
    def _gaus(self, x0, x1):
        return numpy.random.normal(numpy.absolute(x0), numpy.absolute(x1))

    def subscribe(self, node):
        """Register a listener

        Args:
            node (CGPNode): The node which will from now on listen to this node
        """
        # Why can a node have several instances of the same subscriber?
        # Because a node can give it's output to several of another nodes inputs. 
        self.subscribers.append(node)
    
    def alert_subscribers(self):
        for node in self.subscribers:
            node.input_ready()
    
    def genRandomNode(self, existing_nodes, counter, priority_nodes=None, debug=False):
        if priority_nodes:  # For use when favoring some nodes, i.e. swapping out subgraph with random subgraph
            #use_nodes = priority_nodes
            use_nodes = existing_nodes
        else:
            use_nodes = existing_nodes
        node_type = randchoice(self.CPGNodeTypes)
        if node_type in self.twoary:
            if len(use_nodes) < 2:
                if priority_nodes:
                    parent1 = use_nodes[0]
                    parent2 = randchoice(existing_nodes)
            else:
                parent1 = randchoice(use_nodes)
                parent2 = randchoice(use_nodes)
            node = CGPNode(node_type, [parent1, parent2], counter, max((parent1.row_depth, parent2.row_depth))+1, debug)
        elif node_type in self.oneary:
            parent1 = randchoice(use_nodes)
            node = CGPNode(node_type, [parent1], counter, parent1.row_depth+1, debug)
        node.validate()
        return node



class CGPNode(NodeAbstract):
    def __init__(self, node_function_type, inputs, counter, row_depth, debugging=False) -> None:
        """Node for CGP, non-recursive

        Args:
            node_function_type (str or CGPModuleType): One of the defined functions in the global variable CGPNodeTypes
            inputs (List<CGPNode>): List of nodes which are given as the input nodes to this node on creation
            counter (Counter): Counter object to give each node unique ID.
            row_depth (): [description]
            debugging (bool, optional): [description]. Defaults to False.
        """
        # Written to be slightly tolerant of incorrect input arity, but not really
        # Written for non-recursive CGP
        super().__init__()
        self.counter = counter
        self.type = node_function_type
        self.ready_inputs = 0
        self.inputs = inputs
        for input_node in inputs: 
            input_node.subscribe(self)
        self.id = self.counter.counterval()
        self.row_depth = self.id  # Just to get an initial ordering
        self.debugging = debugging

        self.age = 1  # Used for extracting cones
        if type(self.type) is not CGPModuleType:
            self.arity = self.NodeTypeArity[self.type]
        else:
            self.arity = self.type.arity

    def gettype(self):
        """Returns string representation of node type

        Returns:
            str: String grepresentatino of node type
        """
        return_string_parts = []
        if type(self.type) is not CGPModuleType:
            return_string_parts.append(self.type)
        else:
            return_string_parts.append(f"module_{self.type.gettype()}")
        return "   |   ".join(return_string_parts)
    
    def __str__(self) -> str:
        """Returns string grepresentation of node (shorter than gettype)

        Returns:
            str: string representation of node
        """
        if type(self.type) is not CGPModuleType:
            return self.type
        else:
            return f"CGPModule({str(self.type.arity)})"

    def input_ready(self):
        """Increments count of node inputs, if node has enough inputs to execture exectues. 
        """

        # If adding new node functions, take care to implement cropping values to a specific range
        # if new node functions can reasonably cause overflow, ex. multiplication.
        if self.debugging:
            if len(self.inputs) > self.arity:
                raise Exception("Inputs greater than node type arity on input_ready")
            if self.ready_inputs > self.arity:
                raise Exception("More ready inputs than maximum inputs. Likely forgetting to reset node.")
        self.ready_inputs += 1
        if type(self.type) is CGPModuleType and self.ready_inputs == self.arity:
            self.output = self.type.run([x.output for x in self.inputs])
            self.alert_subscribers()    
        elif self.ready_inputs == 2 and type(self.type) is not CGPModuleType and self.NodeTypeArity[self.type] == 2:
            self.output = self.type_func_map[self.type](self.inputs[0].output, self.inputs[1].output)
            self.alert_subscribers()
        elif self.ready_inputs >= 1 and type(self.type) is not CGPModuleType and self.NodeTypeArity[self.type] == 1:
            self.output = self.type_func_map[self.type](self.inputs[0].output)
            self.alert_subscribers()
             

    def change_type(self, newtype):
        """Change the type/node function of the node

        Args:
            newtype (str or CGPModuleType): New node function/type
        """
        self.age = 1
        self.type = newtype
        if type(newtype) is CGPModuleType:
            self.arity = newtype.arity
        else:
            self.arity = self.NodeTypeArity[self.type]
        while len(self.inputs) > self.arity:
            to_remove = self.inputs[randchoice(range(0, self.arity))]
            self.inputs.remove(to_remove)
            to_remove.subscribers.remove(self)


    def reset(self):
        self.ready_inputs = 0
        self.output = None

    
    def add_connection(self, node):
        if len(self.inputs) + 1 > self.arity:
            raise Exception("Trying to add input node when function arity is already met")
        self.inputs.append(node)
        node.subscribe(self)
        self.update_row_depth()

    def update_row_depth(self):
        old_depth = int(self.row_depth)
        if len(self.inputs) == 0:
            self.row_depth = 1
        else:
            self.row_depth = numpy.amax([x.row_depth for x in self.inputs])+1
        if self.row_depth != old_depth:
            for node in self.subscribers:
                node.update_row_depth()
    
    def validate(self):
        # Check that node has a valid state
        if self.debugging:
            # Where None means that it is disconnected/not a valid node
            if self.inputs is not None and len(self.inputs) > self.arity:
                raise Exception("Invalid input arity")
            if self.inputs is not None: 
                for input_node in self.inputs:
                    if self not in input_node.subscribers: 
                        raise Exception("Node not registered in subscribers")
            if self.subscribers is not None: 
                for subscriber in self.subscribers:
                    if self not in subscriber.inputs:
                        raise Exception("Node subscribes to this one without registering it as input")


class InputCGPNode(NodeAbstract):
    def __init__(self, counter, debugging = False):
        self.counter = counter
        super().__init__()
        self.id = self.counter.counterval()
        self.row_depth = 0
        self.debugging = debugging
        self.output = None
        self.type = "InputNode"
        self.arity = 0
        self.age = 1
        self.inputs = []
    
    def set_output(self, float_val):
        self.output = float_val
    
    def validate(self):
        if self.debugging:
            for subscriber in self.subscribers:
                if not self in subscriber.inputs: 
                    raise Exception("Input node subscriber does not have input node as one of its inputs")
    
    def reset(self):
        self.output = None
    
    def gettype(self):
        return "Input node"
    
    def __str__(self) -> str:
        return self.type
    
    def update_row_depth(self):
        pass

def recursive_remove(nodes, remove_from):
    if len(nodes) > 1:
        if nodes[0] in remove_from:
            remove_from.remove(nodes[0])
        recursive_remove(nodes[1:], remove_from)
        return None
    if len(nodes) == 1:
        if nodes[0] in remove_from:
            remove_from.remove(nodes[0])
        return None

class CGPProgram:
    def __init__(self, input_arity, output_arity, counter, config, debugging = False, max_size=None, init_nodes = True) -> None:
        self.config = config
        self.counter = counter
        self.input_arity = input_arity
        self.input_nodes = [InputCGPNode(self.counter, debugging) for _ in range(self.input_arity)]
        self.nodes = []
        self.output_arity = output_arity
        self.debugging = True
        if max_size is None:
            self.max_size = self.config['cgp_program_size']
        else:
            self.max_size = max_size
        self.output_absolute_maxvalue = 10**4

        _dummy_node = InputCGPNode(self.counter, debugging)
        if init_nodes:
            for _ in range(self.output_arity):
                new_node = _dummy_node.genRandomNode(self.input_nodes + self.nodes, counter, None, self.debugging)
                self.nodes.append(new_node)
            for _ in range(self.max_size - self.output_arity):
                node_type = randchoice(_dummy_node.CPGNodeTypes)
                new_node = CGPNode(node_type, [], self.counter, 0, self.debugging)
                self.nodes.append(new_node)
        self.output_indexes = [x for x in range(0, output_arity)]  
        # Because first nodes are guaranteed to be connected at start


    def set_config(self, config):
        self.config = config


    def validate_input_node_array(self):
        if self.debugging and len(self.input_nodes) > self.input_arity:
            raise Exception("Nodes added to input_nodes")


    def run(self, input_vals):
        self.reset()
        #self.validate_nodes()
        if len(input_vals) != self.input_arity:
            raise Exception("Input val length does not match input arity of program")
        for num in range(0, len(input_vals)):
            self.input_nodes[num].set_output(input_vals[num])
            self.input_nodes[num].alert_subscribers()
        #self.validate_input_node_array()
        outputs = []
        for index in self.output_indexes:
            output = self.nodes[index].output
            outputs.append(output)
        if None in outputs:
            raise Exception("None in outputs detected")
        self.reset()
        return outputs


    def get_node_type_counts(self):
        node_type_counts = {}
        for node in self.get_active_nodes():
            if type(node) != InputCGPNode:
                if type(node) != CGPModuleType:
                    if node.type not in node_type_counts.keys():
                        node_type_counts[node.type] = 1
                    else:
                        node_type_counts[node.type] += 1
                else:
                    for key, val in node.program.get_node_type_counts().items():
                        if key not in node_type_counts.keys():
                            node_type_counts[node.type] = val
                        else:
                            node_type_counts[node.type] += val
        return node_type_counts

    

    def run_presetinputs(self):
        if None in [x.output for x in self.input_nodes]:
            raise Exception("All outputs not set")
        for input_node in self.input_nodes:
            input_node.alert_subscribers()
        outputs = []
        for index in self.output_indexes:
            output = self.nodes[index].output
            outputs.append(output)
        return outputs


    def get_active_nodes(self):
        self.reset()
        self_nodes = self.nodes
        for node in self.input_nodes:
            node.set_output(1.0)
            node.alert_subscribers()
        active_nodes = [x for x in self_nodes if x.output is not None]

        output_nodes = [self_nodes[x] for x in self.output_indexes if self_nodes[x] in active_nodes]
        frontier = [x for x in output_nodes]
        active_nodes2 = [x for x in output_nodes]
        while len(frontier) > 0:
            new_frontier = []
            for node in frontier:
                for node2 in node.inputs:
                    if node2 not in new_frontier and node2 in active_nodes:
                        new_frontier.append(node2)
                        active_nodes2.append(node2)
            frontier = new_frontier
        self.reset()
        active_nodes = list(set(active_nodes2))
        return active_nodes

    def get_active_modules(self, recursive=False):
        module_types = []
        depth = 0
        for node in self.get_active_nodes():
            if type(node) != InputCGPNode and type(node.type) is CGPModuleType and node.id not in [x.id for x in module_types]:
                module_types.append(node.type)
                if recursive:
                    new_modules, new_depth = node.type.program.get_active_modules(recursive)
                    if new_depth != 0:
                        depth = new_depth
                    module_types += new_modules
        return module_types, depth + 1

    def simple_mutate(self, cgp_modules = None):
        #self.validate_nodes()
        self_nodes = self.nodes
        input_nodes = self.input_nodes
        for node in self_nodes:
            node.age += 1
        self_config = self.config
        node_link_mutate_chance = self_config['mutation_chance_link']
        node_type_mutate_chance = self_config['mutation_chance_node']
        def _simple_mutate(nodes):
            module_types = []
            if cgp_modules is not None:
                module_types = cgp_modules
            else:
                module_types, _ = self.get_active_modules()
            for node in nodes:
                for num in range(0, node.arity):
                    if randcheck(node_link_mutate_chance):
                        # Normally there is a "maximal output connections" per node paramter too, in this version
                        # this is not implemented
                        input_from = randchoice([x for x in nodes if x.row_depth < node.row_depth] + input_nodes)
                        if len(node.inputs) >= node.arity: # neutral in connection count
                            to_remove = node.inputs[num]
                            node.inputs.remove(to_remove)
                            to_remove.subscribers.remove(node)
                            to_remove.update_row_depth()
                            node.add_connection(input_from)
                        elif randcheck(0.5): # 0.5 chance to add connection
                            node.add_connection(input_from)
                        elif len(node.inputs) > 0: # 0.5 chance to remove
                            if num < len(node.inputs):
                                to_remove = node.inputs[num]
                                node.inputs.remove(to_remove)
                                to_remove.subscribers.remove(node)
                                to_remove.update_row_depth()
                
                # Scale probability of type change with node arity s.t. type drift does not favour
                # low arity functions - with the exception of when node type mutate change is 1.0,
                # i.e. hypermutation or complete randomness
                if randcheck(node_type_mutate_chance/node.arity) or node_type_mutate_chance == 1.0:
                    #node.change_type(randchoice(CPGNodeTypes + module_types))
                    node.change_type(randchoice(self_nodes[0].CPGNodeTypes + module_types))
                node.validate()

        #self.validate_nodes()
        output_node_alternatives = []
        while len(output_node_alternatives) == 0:
            self.reset()
            _simple_mutate(self_nodes)
            #self.validate_input_node_array()
            #self.validate_nodes()
            for node in input_nodes:
                node.set_output(1.0)
                node.alert_subscribers()
            output_node_alternatives = [x for x in self_nodes if x.output is not None]
            self.reset()
        for num in range(0, len(self.output_indexes)):
            if random.random() < node_type_mutate_chance or self_nodes[self.output_indexes[num]] not in output_node_alternatives:
                self.output_indexes[num] = self_nodes.index(randchoice(output_node_alternatives))
        #self.validate_input_node_array()
        return self
    
    def validate_nodes(self):
        if self.debugging:
            for node in self.nodes:
                node.validate()
    
    def get_input_none_index(self):
        index = 0
        for input_node in self.input_nodes:
            if input_node.output is None:
                return index
            index += 1
        raise Exception("None not found in input nodes. Make sure to reset program.")

    def deepcopy(self):
        # Returns deep copy of self
        #self.validate_nodes()
        new_copy = CGPProgram(self.input_arity, self.output_arity, self.counter, self.config, self.debugging, False)
        input_node_copy = []
        for input_node in self.input_nodes:
            new_node = InputCGPNode(self.counter, self.debugging)
            new_node.id = input_node.id
            input_node_copy.append(new_node)
        node_copy = []
        #self.validate_nodes()
        for node in self.nodes:
            new_node = CGPNode(node.type, [], self.counter, 0, self.debugging)
            new_node.id = node.id
            new_node.row_depth = node.row_depth+1
            new_node.arity = node.arity
            node_copy.append(new_node)
        #self.validate_nodes()
        input_ids = [x.id for x in input_node_copy]
        node_ids = [x.id for x in node_copy]
        for node in input_node_copy:
            input_index = input_ids.index(node.id)
            target_ids = [x.id for x in self.input_nodes[input_index].subscribers]
            target_indexes = [node_ids.index(x) for x in target_ids]
            for target in target_indexes:
                node_copy[target].add_connection(node)
            node.validate()
        for node in node_copy:
            input_index = node_ids.index(node.id)
            target_ids = [x.id for x in self.nodes[input_index].inputs]
            target_indexes = [(node_ids + input_ids).index(x) for x in target_ids]
            for target in target_indexes:
                target = (node_copy + input_node_copy)[target]
                # If it is a connection has already been established above
                if not type(target) == InputCGPNode:
                    node.add_connection(target)
            node.validate()
        

        new_copy.input_nodes = input_node_copy
        new_copy.nodes = node_copy
        new_copy.output_indexes = [x for x in self.output_indexes]
        #self.validate_nodes()
        new_copy.validate_nodes()
        return new_copy
        

    def produce_children(self, child_count, cgp_modules = None):
        alternatives = [self.deepcopy() for num in range(0, child_count)]
        alternatives = [x.simple_mutate(cgp_modules) for x in alternatives]
        return alternatives


    def reset(self):
        for node in self.nodes + self.input_nodes:
            node.reset()
    
    
    def extract_subgraphs(self, max_size, subgraph_count, node_pool):
        """
        Extracts count subgraphs from program in size range [1, max_size]. May be overlapping. Returns
        CGPModuleTypes.
        """
        node_pool = [x for x in node_pool if type(x) is not InputCGPNode]  # Bias towards things we know are working
        subgraphs = []
        if not len(node_pool) == 0:
            while len(subgraphs) < subgraph_count:
                target_size = randchoice(range(max_size))
                subgraph_nodes = []
                origin_node = randchoice_scaled(node_pool, [x.age for x in node_pool])
                subgraph_nodes += [origin_node]
                frontier = [x for x in origin_node.inputs if type(x) is not InputCGPNode]
                if origin_node in frontier:
                    raise Exception("Critical error auto-recursive loop in CGP graph")

                # to avoid infinite loops
                max_tries = 100
                current_try = 0
                while len(frontier) > 0 and len(subgraph_nodes) < target_size and current_try < max_tries:
                    selected_node = randchoice_scaled(frontier, [x.age for x in frontier])
                    old_frontier = [x for x in frontier]
                    frontier.remove(selected_node)
                    
                    # As pointed out in "Advanced techniques for the creation and propagation of modules in cartesian genetic programming"
                    # by Kaufmann & Platzner a reconvergent connection with a cone module is when a node in the cone gives an input to 
                    # a node outside of the cone which gives an input to the cone again, causing a in principle endless recurrence.
                    reconvergent_path_check = False
                    if len(frontier) > 0:
                        for node in frontier:
                            if selected_node in node.inputs:
                                reconvergent_path_check = True
                                break
                    if not reconvergent_path_check:
                        for node in selected_node.inputs:
                            if node not in subgraph_nodes and node not in frontier and \
                                    type(node) != InputCGPNode:
                                frontier.append(node)
                        if selected_node not in subgraph_nodes:
                            subgraph_nodes += [selected_node]
                    else: 
                        frontier = old_frontier
                    current_try += 1

                subgraph_partly_copied = []
                # Init copies
                for node in subgraph_nodes:
                    copy_node = CGPNode(node.type, [], node.counter, node.row_depth, node.debugging)
                    copy_node.id = node.id
                    copy_node.arity = node.arity
                    subgraph_partly_copied += [copy_node]
                # Connect copies
                subgraph_input_arity = 0
                subgraph_node_ids = [x.id for x in subgraph_partly_copied]
                if self.debugging:
                    for id in subgraph_node_ids:
                        c = count([x for x in subgraph_node_ids if x == id])
                        if c > 1:
                            raise Exception("Duplicate ID in subgraph node ids")
                for node in subgraph_nodes:
                    for input_node in node.inputs:
                        if input_node.id in subgraph_node_ids:
                            input_from = subgraph_partly_copied[subgraph_node_ids.index(input_node.id)]
                            input_to = subgraph_partly_copied[subgraph_node_ids.index(node.id)]
                            input_to.add_connection(input_from)
                        else:
                            subgraph_input_arity += 1
                subgraphs.append(CGPModuleType(subgraph_input_arity, subgraph_partly_copied, 0, self.counter, self.config, self.debugging))
            return subgraphs
    
    
    def __eq__(self, o: object) -> bool:
        ownnodes = self.input_nodes + self.nodes
        onodes = o.input_nodes + o.nodes
        if not len(ownnodes) == len(onodes):
            return False
        elif self.input_arity != o.input_arity:
            return False
        elif [x.id for x in ownnodes] != [x.id for x in onodes]:
            return False
        elif self.output_indexes != o.output_indexes:
            return False
        return True
    
    def __str__(self) -> str:
        return "-".join([str(x) for x in self.input_nodes]) + "-" + "-".join([str(x) for x in self.nodes]) + "-".join([str(x) for x in self.output_indexes])


def subgraph_crossover(mate1, mate2, subgraph_extract_count, subgraph_size):
    mate1_active_nodes = mate1.get_active_nodes()
    mate2_active_nodes = mate2.get_active_nodes()
    mate1_inactive_nodes = [x for x in mate1.nodes if x not in mate1_active_nodes]
    mate2_inactive_nodes = [x for x in mate2.nodes if x not in mate2_active_nodes]
    mate1_subgraphs = mate1.extract_subgraphs(subgraph_size, min(subgraph_extract_count, len(mate1_active_nodes)),mate1_active_nodes)
    mate2_subgraphs = mate2.extract_subgraphs(subgraph_size, min(subgraph_extract_count, len(mate2_active_nodes)),mate2_active_nodes)
    for subgraph in mate1_subgraphs:
        if len(mate2_inactive_nodes) != 0:
            target = randchoice(mate2_inactive_nodes)
        else:
            target = randchoice(mate2.nodes)
        target.change_type(subgraph)
    for subgraph in mate2_subgraphs:
        if len(mate1_inactive_nodes) != 0:
            target = randchoice(mate1_inactive_nodes)
        else:
            target = randchoice(mate1.nodes)
        target.change_type(subgraph)


class CGPModuleType:
    def __init__(self, arity, nodes, output_node_index, module_counter, config,debugging=False) -> None:
        self.id = module_counter.counterval()
        self.config = config
        self.arity = arity
        self.nodes = nodes
        self.program = CGPProgram(arity, 1, Counter(), self.config, False, max_size = arity+1, init_nodes=False)
        self.program.nodes = self.nodes
        self.output_indexes = [output_node_index]
        free_inputs = arity
        connected_input_nodes = 0
        while free_inputs - connected_input_nodes > 0:
            for node in self.program.nodes:
                if len(node.inputs) < node.arity:
                    node.add_connection(self.program.input_nodes[connected_input_nodes])
                    connected_input_nodes += 1
        self.program.validate_nodes()
        self.debugging = debugging
        if debugging: 
            if self.arity != self.program.input_arity:
                raise Exception("Module-program arity mismatch")
    
    def run(self, input_vals):
        return self.program.run(input_vals)[0]

    def gettype(self):
        return self.id
