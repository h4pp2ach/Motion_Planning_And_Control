import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_1 import map

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True
                
def get_action():
    # action = [dx, dy, cost]
    # action_set = [action1, action2, ...]
    return action_set

def collision_check(omap, node):
    # Check if node position == obstacle position
    return col # True or False

def dijkstra(start, goal, map_obstacle):
    
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    
    open_list = []
    closed_list = []
    
    open_list.append(start_node)
    while open_list is not None:
        # Find node with lowest cost
                
        # If goal, return optimal path
        
        # If not goal, move from open list to closed list

        # Generate child candidate
        action_set = get_action()
        for action in action_set:
            # If collision expected, do nothing

            # If not collision, create child node

            # If already in closed list, do nothing

            # If not in closed list, update open list

        # show graph
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.1)
                
                

def main():

    start, goal, omap = map()

    if show_animation == True:
        plt.figure(figsize=(8,8))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k',  markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = dijkstra(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)
    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "m.-")
        plt.show()


if __name__ == "__main__":
    main()

    

