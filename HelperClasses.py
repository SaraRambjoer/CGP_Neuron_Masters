import random
import math

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

def randcheck(val):
    return random.random() < val