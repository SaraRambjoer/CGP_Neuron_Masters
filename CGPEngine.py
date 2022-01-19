import math
import random
from HelperClasses import Counter, randchoice, randcheck, randchoice_scaled
from numpy.core.fromnumeric import sort, var
from numpy.lib.function_base import average
from numpy.ma import count

# IDEA for functions: 
# common math operations, then threshold units giving binary output, then logic gate stuff
# possible actions are 1's and 0's, maybe except changing internal state variables
# inputs should also be common numbers, i.e. 1, 2, 10

CPGNodeTypes = [
    "ADDI",
    "SUBI",
    "MULT",
    "DIVI",
    "SINU",
    "NOT",
    "GTE",
    "ISZERO",
    "AND",
    "NAND",
    "XOR",
    "OR",
    "EQ",
    "ROUND"  # TO 1 or 0
]

NodeTypeArity = {
    "ADDI":2,
    "SUBI":2,
    "MULT":2,
    "DIVI":2,
    "SINU":1,
    "NOT":1,
    "GTE":2,
    "ISZERO":1,
    "OR":2,
    "AND":2,
    "NAND":2,
    "XOR":2,
    "EQ":2,
    "ROUND":1
}
oneary = [x[0] for x in NodeTypeArity.items() if x[1] == 1]
twoary = [x[0] for x in NodeTypeArity.items() if x[1] == 2]

class NodeAbstract():
    def __init__(self) -> None:
        self.subscribers = []
        self.output = None

    def subscribe(self, node):
        # Why can a node have several instances of the same subscriber?
        # Because a node can give it's output to several of another nodes inputs. 
        self.subscribers.append(node)
    
    def alert_subscribers(self):
        for node in self.subscribers:
            node.input_ready()


class CGPNode(NodeAbstract):
    def __init__(self, node_function_type, inputs, counter, row_depth, debugging=False) -> None:
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

        self.age = 0  # Used for extracting cones
        if type(self.type) is not CGPModuleType:
            self.arity = NodeTypeArity[self.type]
        else:
            self.arity = self.type.arity

    def gettype(self):
        return_string_parts = []
        if type(self.type) is not CGPModuleType:
            return_string_parts.append(self.type)
        else:
            return_string_parts.append(f"module_{self.type.gettype()}")
        return "   |   ".join(return_string_parts)
    
    def __str__(self) -> str:
        if type(self.type) is not CGPModuleType:
            return self.type
        else:
            return f"CGPModule({str(self.type.arity)})"

    def input_ready(self):
        if len(self.inputs) > self.arity:
            raise Exception("Inputs greater than node type arity on input_ready")
        if self.ready_inputs > self.arity:
            raise Exception("More ready inputs than maximum inputs. Likely forgetting to reset node.")
        self.ready_inputs += 1
        if type(self.type) is CGPModuleType and self.ready_inputs == self.arity:
            self.output = self.type.run([x.output for x in self.inputs])
        elif self.ready_inputs == 2 and self.type in twoary:
            x0 = self.inputs[0].output
            x1 = self.inputs[1].output
            if self.type == "ADDI":
                self.output = x0 + x1
            elif self.type == "SUBI":
                self.output = x0 - x1
            elif self.type == "MULT":
                self.output = x0 * x1
            elif self.type == "DIVI":
                # To handle very low values to avoid infinity values
                if x1 > -0.1 and x1 < 0.1:
                    if x1 < 0:
                        x1 = -0.1
                    else:
                        x1 = 0.1
                self.output = x0 / x1
            elif self.type == "GTE":
                self.output = 1 if x1 > x0 else 0.0
            elif self.type == "EQ":
                self.output = 1 if x1 == x0 else 0.0
            elif self.type == "AND":
                self.output = 1.0 if x0 == 1.0 and x1 == 1.0 else 0.0
            elif self.type == "NAND":
                self.output = 1.0 if x0 != x1 else 0.0
            elif self.type == "XOR":
                self.output = 1.0 if x0 != x1 and (x0 == 1.0 or x1 == 1.0) else 0.0
            elif self.type == "OR":
                self.output = 1.0 if x0 == 1.0 or x1 == 1.0 else 0.0
            self.alert_subscribers()
        elif self.ready_inputs >= 1 and self.type in oneary:
            x0 = self.inputs[0].output
            if self.type == "SINU":
                if x0 >= float('inf')-1 or x0 >= 999999999999:
                    self.output = 0.0
                else:
                    try:
                        self.output = math.sin(x0)
                    except ValueError:  # For some reason the above doesn't always catch inf, so...
                        self.output = 0
            elif self.type == "ROUND":
                self.output = 1.0 if x0 >= 0.5 else 0.0
            elif self.type == "ISZERO":
                self.output = 1.0 if x0 == 0.0 else 0.0
            elif self.type == "NOT":
                try:
                    self.output = max(1.0 - x0, 0.0)
                except:
                    print(x0)
            self.alert_subscribers()
             

    def change_type(self, newtype):
        self.age = 0
        self.type = newtype
        if type(newtype) is CGPModuleType:
            self.arity = newtype.arity
        else:
            self.arity = NodeTypeArity[self.type]
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

    def disconnect(self):
        if not self.inputs is None:
            for ele in self.inputs:
                ele.break_connection(self)
        if not self.subscribers is None:
            for ele in self.subscribers:
                ele.break_connection(self)
        self.inputs = None
        self.subscribers = None
    
    def break_connection(self, node):
        if not self.inputs is None:
            while node in self.inputs:
                self.inputs.remove(node)
        if not self.subscribers is None:
            while node in self.subscribers:
                self.subscribers.remove(node)
    
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
        self.age = 0
        self.inputs = []
    
    def set_output(self, float_val):
        self.output = float_val
    
    def break_connection(self, node):
        if node in self.subscribers:
            self.subscribers.remove(node)
    
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
    
def genRandomNode(existing_nodes, counter, priority_nodes=None, debug=False):
    # TODO This approach to row depth won't work, because several nodes may have same depth...
    # Issue is that when hooking these nodes one may create a loop, so we need to be able to partition
    # the nodes around the node being swapped out.
    # ^^ I think this was fixed by introducing counter unique ids but I can't quite remember lmao
    if priority_nodes:  # For use when favoring some nodes, i.e. swapping out subgraph with random subgraph
        #use_nodes = priority_nodes
        use_nodes = existing_nodes
    else:
        use_nodes = existing_nodes
    node_type = randchoice(CPGNodeTypes)
    if node_type in twoary:
        if len(use_nodes) < 2:
            if priority_nodes:
                parent1 = use_nodes[0]
                parent2 = randchoice(existing_nodes)
        else:
            parent1 = randchoice(use_nodes)
            parent2 = randchoice(use_nodes)
        node = CGPNode(node_type, [parent1, parent2], counter, max((parent1.row_depth, parent2.row_depth))+1, debug)
    elif node_type in oneary:
        parent1 = randchoice(use_nodes)
        node = CGPNode(node_type, [parent1], counter, parent1.row_depth+1, debug)
    node.validate()
    return node

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
    def __init__(self, input_arity, output_arity, counter, config, debugging = True, max_size=None, init_nodes = True) -> None:
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
        if init_nodes:
            for _ in range(self.output_arity):
                new_node = genRandomNode(self.input_nodes + self.nodes, counter, None, self.debugging)
                self.nodes.append(new_node)
            for _ in range(self.max_size - self.output_arity):
                node_type = randchoice(CPGNodeTypes)
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
        self.validate_nodes()
        if len(input_vals) != self.input_arity:
            raise Exception("Input val length does not match input arity of program")
        for num in range(0, len(input_vals)):
            self.input_nodes[num].set_output(input_vals[num])
            self.input_nodes[num].alert_subscribers()
        self.validate_input_node_array()
        outputs = []
        for index in self.output_indexes:
            output = self.nodes[index].output
            
            # without clipping program may crash in some rare instances
            # but it should be like super rare so it shouldn't really ever occur...
            #if output > self.output_absolute_maxvalue:
            #    raise Exception()
            #    output = self.output_absolute_maxvalue
            #elif output < -self.output_absolute_maxvalue:
            #    raise Exception()
            #    output = -self.output_absolute_maxvalue
#
            outputs.append(output)
        if None in outputs:
            raise Exception("None in outputs detected")
        self.reset()
        return outputs

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
        for node in self.input_nodes:
            node.set_output(1.0)
            node.alert_subscribers()
        active_nodes = [x for x in self.nodes if x.output is not None]

        output_nodes = [self.nodes[x] for x in self.output_indexes]
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
        return list(dict.fromkeys(active_nodes2))

    def self_prune(self):
        # Removes every node not connected to an output node from self.
        output_ids = [x.id for x in [self.nodes[indx] for indx in self.output_indexes]]
        active_nodes = []
        # Debugging
        for node in self.nodes:
            node.validate()

        for index in self.output_indexes:
            node = self.nodes[index]
            def recursive_collect_children(node, children):
                if node.id in [x.id for x in children]:
                    return children
                if "inputs" in node.__dict__:
                    for child in node.inputs:
                        recursive_collect_children(child, children)
                children.append(node)
                return children
            active_nodes += recursive_collect_children(node, [])
        
        # Debugging
        self.validate_nodes()

        found_node_ids = [x.id for x in active_nodes]
        to_remove = []
        for node in self.nodes:
            if node.id not in found_node_ids:
                to_remove.append(node)
                node.disconnect()

        self.validate_nodes()
        recursive_remove(to_remove, self.nodes)
        self.validate_nodes()
        new_output_indexes = []
        node_ids = [x.id for x in self.nodes]
        for id in output_ids:
            new_output_indexes.append(node_ids.index(id))
        self.output_indexes = new_output_indexes

    def simple_mutate(self):
        self.validate_nodes()
        for node in self.nodes:
            node.age += 1
        #self.fix_row()
        output_change_chance = 0.1
        subgraph_size_max = 5
        node_swap_chance = 1.0
        node_link_mutate_chance = self.config['mutation_chance_link']
        node_type_mutate_chance = self.config['mutation_chance_node']
        def _simple_mutate_randsubgraph(nodes):
            if random.random() < node_swap_chance:
                node = randchoice(nodes)
                node_inputs = [x for x in node.inputs]
                node_output_links = node.subscribers
                node.disconnect()
                new_nodes = []
                subgraph_size = randchoice(range(1, subgraph_size_max))
                for _ in range(subgraph_size):
                    legal_inputs = [x for x in self.nodes + self.input_nodes if x.row_depth < node.row_depth]
                    new_node = genRandomNode(legal_inputs, self.counter, node_inputs + new_nodes, self.debugging)
                    new_nodes.append(new_node)
                    new_node.validate()
                recursive_remove(node_inputs, new_nodes)
                self.nodes += new_nodes
                for output_node in node_output_links:
                    output_node.add_connection(randchoice(new_nodes))
                self.nodes.remove(node)
                node.validate()
            return None
        def _simple_mutate(nodes):
            effected_nodes = []
            module_types = []
            for node in self.get_active_nodes():
                if type(node) != InputCGPNode and node.type not in CPGNodeTypes and node.id not in [x.id for x in module_types]:
                    module_types.append(node.type)
            for node in nodes:
                if randcheck(node_link_mutate_chance):
                    # Normally there is a "maximal output connections" per node paramter too, in this version
                    # this is not implemented
                    effected_nodes.append(node.id)
                    input_from = randchoice([x for x in (self.nodes + self.input_nodes) if x.row_depth < node.row_depth])
                    if len(node.inputs) >= node.arity: # neutral in connection count
                        while len(node.inputs) >= node.arity:
                            to_remove = node.inputs[randchoice(range(0, node.arity))]
                            node.inputs.remove(to_remove)
                            to_remove.subscribers.remove(node)
                        node.add_connection(input_from)
                    elif randcheck(0.5): # 0.5 chance to add connection
                        node.add_connection(input_from)
                    elif len(node.inputs) > 0: # 0.5 chance to remove
                        to_remove = node.inputs[randchoice(range(0, len(node.inputs)))]
                        node.inputs.remove(to_remove)
                        to_remove.subscribers.remove(node)
                
                # Scale probability of type change with node arity s.t. type drift does not favour
                # low arity functions
                if randcheck(node_type_mutate_chance/node.arity):
                    effected_nodes.append(node.id)
                    #node.change_type(randchoice(CPGNodeTypes + module_types))
                    node.change_type(randchoice(CPGNodeTypes + module_types))
                node.validate()
            return effected_nodes

        self.validate_nodes()
        output_node_alternatives = []
        while len(output_node_alternatives) == 0:
            self.reset()
            _simple_mutate(self.nodes)
            # Keep mutating until genotype change
            #effected_nodes = _simple_mutate(self.nodes)
            #while len([x for x in effected_nodes if x in active_nodes]) == 0:
            #    effected_nodes = _simple_mutate(self.nodes)
            self.validate_input_node_array()
            self.validate_nodes()
            for node in self.input_nodes:
                node.set_output(1.0)
                node.alert_subscribers()
            output_node_alternatives = [x for x in self.nodes if x.output is not None]
            self.reset()
        for num in range(0, len(self.output_indexes)):
            if random.random() < node_type_mutate_chance or self.nodes[self.output_indexes[num]] not in output_node_alternatives:
                self.output_indexes[num] = self.nodes.index(randchoice(output_node_alternatives))
        self.validate_input_node_array()
        # In practice not pruning is better as it allows for multi-step neutral drift 
        #self.self_prune()
        #self.fix_row()  # fix row not necessary for link-only CGP mutation
        return self
    
    def validate_nodes(self):
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
        self.validate_nodes()
        new_copy = CGPProgram(self.input_arity, self.output_arity, self.counter, self.config, self.debugging, False)
        input_node_copy = []
        for input_node in self.input_nodes:
            new_node = InputCGPNode(self.counter, self.debugging)
            new_node.id = input_node.id
            input_node_copy.append(new_node)
        node_copy = []
        self.validate_nodes()
        for node in self.nodes:
            new_node = CGPNode(node.type, [], self.counter, 0, self.debugging)
            new_node.id = node.id
            new_node.row_depth = node.row_depth+1
            new_node.arity = node.arity
            node_copy.append(new_node)
        self.validate_nodes()
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
        self.validate_nodes()
        new_copy.validate_nodes()
        return new_copy
        

    def produce_children(self, child_count):
        alternatives = [self.deepcopy() for num in range(0, child_count)]
        alternatives = [x.simple_mutate() for x in alternatives]
        return alternatives

    def four_plus_one_evolution(self, eval_routine, inputs):
        # Standard 4-child 1 survivor evolution policy for CGP
        alternatives = self.produce_children(4)
        alternatives.append(self)
        outputs = []
        for cgp_program in alternatives:
            output = []
            for input_tuple in inputs:
                output.append(cgp_program.run(input_tuple))
            outputs.append(output)
        eval_scores = [eval_routine(x) for x in outputs]
        max_score = max(eval_scores)
        max_index = eval_scores.index(max_score)
        print(max_score, max_index, len(alternatives[max_index].nodes), self.nodes[0].type)
        if max_index != 4:
            winner = alternatives[max_index]
            self.nodes = winner.nodes
            self.input_nodes = winner.input_nodes
            self.output_indexes = winner.output_indexes
        self.reset()
        # Max should default to the first found, so since we put the original in the back, the CGP should neutral drift. 

    def reset(self):
        for node in self.nodes + self.input_nodes:
            node.reset()
    
    def fix_row(self):
        # When setting row depth on creation what is actually being set is an estimate of row depth (edges from leaf).
        # This function fixes this and places one node on each row
        # Additionally fixes changes which may occur from changing the network. 
        # RFI Make updates when changing network automatic to improve runtime
        visited_nodes = []
        frontier = self.input_nodes
        depth = 1
        bins = []
        bins.append(self.input_nodes)
        # Sort nodes into bins based on depth from first node
        while len(frontier) > 0:
            new_frontier = []
            for node in frontier:
                for subscriber in node.subscribers:
                    # To ensure that maximal depth is used as basis for assigning row
                    # i.e. a node may be directly connected to an input node, but it should still be possible for that node to have
                    # many nodes calculating its other input
                    if subscriber in visited_nodes and subscriber.row_depth != depth:
                        bins[subscriber.row_depth].remove(subscriber)
                    visited_nodes.append(subscriber)
                    new_frontier.append(subscriber)
                    subscriber.row_depth = depth
            bins.append(new_frontier)
            depth += 1
            frontier = new_frontier
        
        row = 1
        for bin in bins: 
            for node in bin:
                node.row_depth = row
                row += 1
        
        self.validate_acyclic()
    
    def validate_acyclic(self):
        # The following function checks that the CGP program is acyclic, and if it is not it throws an exception. 
        # The function is not necessary for the program to run, but is helpful for debugging new code. 
        if self.debugging:
            for node in self.nodes:
                for subscriber in node.subscribers: 
                    if subscriber.row_depth < node.row_depth:
                        raise Exception("Cyclic CGP program detected")

    def extract_subgraphs(self, max_size, subgraph_count):
        """
        Extracts count subgraphs from program in size range [1, max_size]. May be overlapping. Returns
        CGPModuleTypes.
        """
        node_pool = [x for x in self.get_active_nodes() if type(x) is not InputCGPNode]  # Bias towards things we know are working
        subgraphs = []
        if not len(node_pool) == 0:
            while len(subgraphs) < subgraph_count:
                target_size = randchoice(range(max_size))
                subgraph_nodes = []
                origin_node = randchoice_scaled(node_pool, [x.age for x in node_pool])
                subgraph_nodes += [origin_node]
                frontier = [origin_node.inputs]

                while len(frontier) > 0 and len(subgraph_nodes) < target_size:
                    selected_node = randchoice_scaled(frontier, [x.age for x in node_pool])
                    old_frontier = [x for x in frontier]
                    frontier = frontier.remove(selected_node)
                    
                    # As pointed out in "Advanced techniques for the creation and propagation of modules in cartesian genetic programming"
                    # by Kaufmann & Platzner a reconvergent connection with a cone module is when a node in the cone gives an input to 
                    # a node outside of the cone which gives an input to the cone again, causing a in principle endless recurrence.
                    reconvergent_path_check = False
                    for node in frontier:
                        if selected_node in node.inputs:
                            reconvergent_path_check = True
                            break
                    if not reconvergent_path_check:
                        for node in selected_node.inputs:
                            if node not in subgraph_nodes and node not in frontier and \
                                    type(node) != InputCGPNode:
                                frontier.append(node)
                        subgraph_nodes += [selected_node]
                        frontier.remove(selected_node)
                    else: 
                        frontier = old_frontier
                
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
    
    def eval(self, inputs, eval_func):
        output = []
        for input_tuple in inputs:
            output.append(self.run(input_tuple))
        eval_score = eval_func(output)
        return eval_score
    
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
    mate1_subgraphs = mate1.extract_subgraphs(subgraph_size, min(subgraph_extract_count, len(mate2_active_nodes)))
    mate2_subgraphs = mate1.extract_subgraphs(subgraph_size, min(subgraph_extract_count, len(mate1_active_nodes)))

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


class EvolutionController():
    def __init__(self, population_size, child_count, input_arity, output_arity, debugging, subgraph_extract_count, subgraph_max_size):
        if population_size % 2 == 1:
            raise Exception("Population size should be an even number for mating")
        # init population
        self.population = [CGPProgram(input_arity, output_arity, Counter(), self.config, debugging) for x in range(population_size)]
        self.child_count = child_count
        self.population_size = population_size
        self.debugging = debugging
        self.max_size = self.config['cgp_program_size']
        self.population_fitness = [None for x in range(population_size)]
        self.subgraph_extract_count = subgraph_extract_count
        self.subgraph_max_size = subgraph_max_size
    
    def evolution_step(self, inputs, eval_func):
        if None in self.population_fitness:
            for num in range(0, len(self.population)):
                self.population_fitness[num] = self.population[num].eval(inputs, eval_func)
        egligable_bachelors = [pop for pop in self.population]
        # This is a stupid mating strategy, so room for improvement here, but I suppose it could
        # work okay with neutral drift and elitism. 
        # Should also be quite quick
        while len(egligable_bachelors) > 0:
            mate1 = randchoice(egligable_bachelors)
            egligable_bachelors.remove(mate1)
            mate2 = randchoice(egligable_bachelors)
            egligable_bachelors.remove(mate2)
            self.crossover(mate1, mate2)
        # Then produce new chlidren 
        children = []
        for parent in self.population:
            children += parent.produce_children(self.child_count)
        children_fitness = [child.eval(inputs, eval_func) for child in children]
        # Generational selection
        gen = self.population + children
        gen_fitness = self.population_fitness + children_fitness
        gen_data = list(zip(gen, gen_fitness))
        # This could be done faster, currently it is O(n*log(n)), 
        # which would be really bad for large population sizes
        gen_data.sort(key=lambda x: x[1], reverse=True)
        self.population = [x[0] for x in gen_data[:self.population_size]]
        self.fitness = [x[1] for x in gen_data[:self.population_size]]
        print(max(self.fitness), average(self.fitness), var(self.fitness),  # To get a gauge of variance in the population and performance
                    len([x for x in self.population[0].nodes if type(x.type) is CGPModuleType]),  # To see how often modules are in genome of best solution
                    len([x for x in self.population[0].get_active_nodes() if type(x.type) is CGPModuleType]),   # To see how often modules are actually used in best solutoin
                    len(self.population[0].get_active_nodes())  # To see complexity of best solution
                    )
    
    def crossover(self, mate1, mate2):
        # Code duplication issues
        subgraph_crossover(mate1, mate2, self.subgraph_extract_count, self.subgraph_max_size)
        
        






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
