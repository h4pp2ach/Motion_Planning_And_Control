import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_2 import map

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True
                
