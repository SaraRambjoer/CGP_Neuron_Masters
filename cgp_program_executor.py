import numpy

def calc_function(function, init_data, function_input_pointers):
    if function == "SINU":
        return numpy.sin(numpy.absolute(init_data[function_input_pointers[0]]))
    elif function == "ADDI":
        return init_data[function_input_pointers[0]] + init_data[function_input_pointers[1]]
    elif function == "GAUS":
        return numpy.random.normal(numpy.absolute(init_data[function_input_pointers[0]]), numpy.absolute(init_data[function_input_pointers[1]))


def get_func_input_length(function):
    if function == "SINU":
        return 1
    else:
        return 2

# Assumes input is zero padded to max length
def vm_run(instructions, inputs, instruction_length, input_length):


    init_data = inputs
    next_data = [0 for x in range(input_length)]

    function_input_pointers = [0, 0]  # Must be max length of function input possible

    current_level = 0
    func_input_length = 0
    function = ""
    output_data_index = 0 
    function_output = 0.0

    i0 = 0

    while i0 < instruction_length:
        level = instructions[i0]  # int
        i0 += 1
        if level != current_level:
            current_level = level
            init_data = next_data
            next_data = [0 for x in range(input_length)]
        function = instructions[i0]  # string
        i0 += 1
        if function == "MODULAR":
            function_output, i0_passed = vm_run(instructions[i0:], init_data, len(instructions[i0:], input_length))
            i0 += i0_passed
            output_data_index = instructions[i0]  # int
            i0 += 1
            next_data[output_data_index] = function_output
            # Recursively call vm_run with next instruction as first, and input_length as data, modular will be unpacked for compilation
            # then get result, which is function output, and put it at index to next_data given by the next instruction which is that pointer
            # also + returned i0 to our i0 to skip ahead to right point in instruction track
        elif function == "ENDMODULAR":
            return next_data[0], i0
        else: 
            func_input_length = get_func_input_length(function)
            for i1 in range(func_input_length):
                function_input_pointers[i1] = instructions[i0]
                i0 += 1
            output_data_index = instructions[i0]
            i0 += 1
            function_output = calc_func(function, init_data, function_input_pointers)
            next_data[output_data_index] = function_output
        
    return next_data, i0
        
def run_code(instructions, inputs):
    instruction_length = len(instructions)
    input_length = len(inputs)
    return vm_run(instructions, inputs, instruction_length, input_length)
