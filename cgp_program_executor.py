import numpy

def calc_function(function, init_data, function_input_pointers):
    if function == "SINU":
        return numpy.sin(init_data[function_input_pointers[0]])
    elif function == "ADDI":
        return init_data[function_input_pointers[0]] + init_data[function_input_pointers[1]]
    elif function == "GAUS":
        return numpy.random.normal(
            numpy.absolute(
                init_data[
                    function_input_pointers[0]]
                ), 
            numpy.absolute(
                init_data[
                    function_input_pointers[1]]
                )
            )
    elif function == "SUBI":
        return init_data[function_input_pointers[0]] - init_data[function_input_pointers[1]]
    
    elif function == "MULI": 
        result = init_data[function_input_pointers[0]]*init_data[function_input_pointers[1]]
        if result < -100.0:
            return -100.0
        elif result > 100.0:
            return 100.0
        return result
    
    elif function == "DIVI":
        if abs(init_data[function_input_pointers[1]]) < 0.01:
            if init_data[function_input_pointers[1]] < 0.0:
                result = init_data[function_input_pointers[0]]/-0.01
            else:
                result = init_data[function_input_pointers[0]]/0.01
            if result < -100.0:
                return -100.0
            elif result > 100.0:
                return 100.0
            return result
        return init_data[function_input_pointers[0]]/init_data[function_input_pointers[1]]


def get_func_input_length(function):
    if function == "SINU":
        return 1
    elif function in ["ADDI", "GAUS", "SUBI", "MULI", "DIVI"]:
        return 2
    else:
        raise Exception("Function is invalid function", function)

# Assumes input is zero padded to max length
def vm_run(instructions, inputs, instruction_length, input_length):


    data = [x for x in inputs]

    function_input_pointers = [0, 0]  # Must be max length of function input possible

    func_input_length = 0
    function = ""
    output_data_index = 0 
    function_output = 0.0

    i0 = 0
    while i0 < instruction_length:
        function = instructions[i0]  # string
        i0 += 1
        if function == "MODULAR":
            modular_data_width = instructions[i0]
            i0 += 1
            modular_inputs = [0 for _ in range(modular_data_width)]        
            x0 = 0    
            while instructions[i0] != "MODULARINPUTEND":
                modular_inputs[x0] = instructions[i0]
                i0 += 1
                x0 += 1
            i0 += 1
            function_output, i0_passed = vm_run(instructions[i0:], modular_inputs, len(instructions[i0:]), input_length)
            i0 += i0_passed
            output_data_index = instructions[i0]  # int
            i0 += 1
            data[output_data_index] = function_output
            # Recursively call vm_run with next instruction as first, and input_length as data, modular will be unpacked for compilation
            # then get result, which is function output, and put it at index to next_data given by the next instruction which is that pointer
            # also + returned i0 to our i0 to skip ahead to right point in instruction track
        elif function == "ENDMODULAR":
            return data[0], i0
        else: 
            func_input_length = get_func_input_length(function)
            for i1 in range(func_input_length):
                function_input_pointers[i1] = instructions[i0]
                i0 += 1
            output_data_index = instructions[i0]
            i0 += 1
            function_output = calc_function(function, data, function_input_pointers)
            data[output_data_index] = function_output
    
    return data, i0
        
def run_code(instructions, inputs):
    instruction_length = len(instructions)
    input_length = len(inputs)
    return vm_run(instructions, inputs, instruction_length, input_length)
