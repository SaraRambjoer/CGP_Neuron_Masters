import math
import numpy

def cap(val):
    if val < -100.0:
        return -100.0
    elif val > 100.0:
        return 100.0
    return val

def calc_function(function, init_data, function_input_pointers):
    if function == "SINU":
        to_return = numpy.sin(init_data[function_input_pointers[0]])
    elif function == "ADDI":
        to_return = init_data[function_input_pointers[0]] + init_data[function_input_pointers[1]]
    elif function == "GAUS":
        to_return = numpy.random.normal(
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
        to_return = init_data[function_input_pointers[0]] - init_data[function_input_pointers[1]]
    
    elif function == "MULI": 
        to_return = init_data[function_input_pointers[0]]*init_data[function_input_pointers[1]]
    
    elif function == "DIVI":
        if abs(init_data[function_input_pointers[1]]) < 0.01:
            if init_data[function_input_pointers[1]] < 0.0:
                to_return = init_data[function_input_pointers[0]]/-0.01
            else:
                to_return = init_data[function_input_pointers[0]]/0.01
        else:
            to_return = init_data[function_input_pointers[0]]/init_data[function_input_pointers[1]]
    
    # miller functions below

    elif function == "abs":
        to_return = abs(init_data[function_input_pointers[0]])
    elif function =="sqr":
        to_return = init_data[function_input_pointers[0]]*init_data[function_input_pointers[0]]
    elif function == "sqrt":
        to_return = math.sqrt(init_data[function_input_pointers[0]])
    elif function == "cube":
        to_return = init_data[function_input_pointers[0]]*init_data[function_input_pointers[0]]*init_data[function_input_pointers[0]]
    elif function == "exp":
        z0 = init_data[function_input_pointers[0]]
        to_return = (2*math.pow(math.e, (z0+1)) - math.pow(math.e, 2) - 1)/(math.pow(math.e, 2) - 1)
    elif function == "sin":
        to_return = math.sin(init_data[function_input_pointers[0]])
    elif function == "cos":
        to_return = math.cos(init_data[function_input_pointers[0]])
    elif function == "tanh":
        to_return = math.tanh(init_data[function_input_pointers[0]])
    elif function == "inv":
        to_return = - init_data[function_input_pointers[0]]
    elif function == "step":
        to_return = 0.0 if init_data[function_input_pointers[0]] < 0.0 else 1.0
    elif function == "hyp":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = math.sqrt((z0*z0 + z1*z1)/2)
    elif function == "add":
        to_return = (init_data[function_input_pointers[0]] + init_data[function_input_pointers[1]])/2
    elif function == "sub":
        to_return = (init_data[function_input_pointers[0]] - init_data[function_input_pointers[1]])/2
    elif function == "mult":
        to_return = init_data[function_input_pointers[0]]*init_data[function_input_pointers[1]]
    elif function == "max":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = z0 if z0 >= z1 else z1
    elif function == "min":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = z0 if z0 <= z1 else z1
    elif function == "and":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = 1.0 if (z0 > 0.0 and z1 > 0.0) else 0.0
    elif function == "or":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = 1.0 if (z0 > 0.0 or z1 > 0.0) else 0.0
    elif function == "rmux":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        z2 = init_data[function_input_pointers[2]]
        to_return = z0 if z2 > 0.0 else z1
    elif function == "imult":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = - z0*z1
    elif function == "xor":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        to_return = -1.0 if ((z0 > 0.0 and z1 > 0.0) or (z0 < 0.0 and z1 < 0.0)) else 1.0
    elif function == "istep":
        z0 = init_data[function_input_pointers[0]]
        to_return = 0.0 if z0 < 1.0 else -1.0
    elif function == "tand":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        if z0 > 0.0 and z1 > 0.0:
            to_return = 1.0
        elif z0 < 0.0 and z1 < 0.0:
            to_return = -1.0
        else:
            to_return = 0.0
    elif function == "tor":
        z0 = init_data[function_input_pointers[0]]
        z1 = init_data[function_input_pointers[1]]
        if z0 > 0.0 and z1 > 0.0:
            to_return = 1.0
        elif z0 < 0.0 and z1 < 0.0:
            to_return = -1.0
        else:
            to_return = 0.0
    return cap(to_return)  # millers functions are defined over [-1.0, 1.0], but, this is slightly different when you have writable internal states which can go outside interval, so cap is not at that interval




def get_func_input_length(function):
    if function in ["SINU", "abs", "sqrt", "sqr", "cube", "exp", "sin", "cos", "tanh", "inv", "step", "istep"]:
        return 1
    elif function in ["ADDI", "GAUS", "SUBI", "MULI", "DIVI", "hyp", "add", "sub", "mult", "max", "min", "and", "or", "imult", "xor", "tand", "tor"]:
        return 2
    elif function in ["rmux"]:
        return 3
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
