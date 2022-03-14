import configparser
from MainController import process_config

config_filename = "config_complex_4"
config = configparser.ConfigParser()
config.read(config_filename + '.ini')
print(config_filename + '.ini')
config = config["Default"]
config = process_config(config)


neuron_internal_states = config['neuron_internal_state_count']
dendrite_internal_states = config['axon_dendrite_internal_state_count']
signal_dimensionality = config['signal_dimensionality']
dimensions = 3  # other dimensions not supported - code in engine.py specific to 3d grid
hox_variant_count = config['hox_variant_count']
genome_count = config['genome_count']


neuron_function_arities = [
    [1 + dimensions+ neuron_internal_states + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+neuron_internal_states],  # axon birth
    [dimensions + signal_dimensionality + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + signal_dimensionality + neuron_internal_states],  # signal axon
    [dimensions + signal_dimensionality + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + neuron_internal_states+signal_dimensionality],  # recieve signal axon
    [1 + dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2 + neuron_internal_states],  # reciee reward
    [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 7+neuron_internal_states],  # move
    [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2+neuron_internal_states],  # die
    [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 2+neuron_internal_states*2],  # neuron birth
    [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), 9],  # action controller
    [dimensions + neuron_internal_states + len(config['cgp_function_constant_numbers']), hox_variant_count]  # hox selection
]

dendrite_function_arities = [
    [dimensions + signal_dimensionality + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 2+signal_dimensionality+dendrite_internal_states],
    [dimensions + signal_dimensionality + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 2+signal_dimensionality+dendrite_internal_states],
    [dimensions + signal_dimensionality + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+dendrite_internal_states],
    [dimensions + signal_dimensionality + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 4+signal_dimensionality+dendrite_internal_states],
    [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 2+dendrite_internal_states], # Accept connection
    [dimensions + dendrite_internal_states + dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 1], # Break connection
    [1 + dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 2 + dendrite_internal_states], # recieve reward
    [dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 1+signal_dimensionality], # die
    [dimensions + dendrite_internal_states + len(config['cgp_function_constant_numbers']), 3]
]
def factorial(num):
    product = 1
    for num2 in range(1, num):
        product = num2 * product
    return product

def search_space_one_func(config, arity):
    Z = config['cgp_program_size']
    T = 2  # for standards...
    M = arity[1]
    N = arity[0]
    Q = 6
    term1 = 2**(int((Z*factorial(T))//2))
    term2 = factorial(Z)//(factorial(Z-M))
    term3 = Q*Z*Z*factorial(T)
    term4 = factorial(N)//(factorial(N-T))
    return term1*term2*term3*term4

arities = neuron_function_arities + dendrite_function_arities


sizes = []
for num in range(len(arities)):
    arity = arities[num]
    if num != 8: # 8 is hex function selection variant which only has one program at all times
        for hex in range(config['hox_variant_count']):
            sizes.append(search_space_one_func(config, arity))
    else:
        sizes.append(search_space_one_func(config, arity))

for x in sizes:
    print("{:.2e}".format(x))

print("-------")


stringe = ""
for x in sizes: 
    y = "{:.2e}".format(x)
    y = y.split("e+")
    stringe += y[0] + "^" + y[1] + "*"

print("---------")


prod = 1
for num in sizes:
    print(num)
    prod = prod * num

to_print = ""
prod = str(prod)
to_print += prod[0] + "." + prod[1:2] + "e+" + str(len(prod)-1)
print(to_print)


