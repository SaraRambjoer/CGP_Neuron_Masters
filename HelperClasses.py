import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import time
from threading import Lock


class Counter:
    def __init__(self) -> None:
        self.count = 0
        self.counter_lock = Lock()

    def counterval(self):
        self.counter_lock.acquire()
        self.count += 1
        to_return = int(self.count)
        self.counter_lock.release()
        return self.count


# TODO I think the reason that randomness is not correctly seeded is that the correct seed is never set in these helper functions. 
# Solution would be to put these in a class object, such that they can be seeded, and passing around the class object. 
# But to be honest, I think it is a better idea to just forego the idea of proper seeding, and rather focus on multicore parallell
# evaluation of genomes - which would introduce a resource conflict on seeding anyway, unless one want's to reseed the randomness
# for each time a genome is evaluated. Reseeding seems dumb, as the program may evolve to take advantage of the deterministic
# random outcomes in that case. s
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

def dict_merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            dict_merge(value, node)
        else:
            destination[key] = value

    return destination