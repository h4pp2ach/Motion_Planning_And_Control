import numpy as np
import matplotlib.pyplot as plt
from map_4 import map


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class RRT(object):
    def __init__(self, start, goal, space, obstacle_list, success_dist_thres=1.0):
        self.start_node = Node(start[0], start[1])  # node (x, y)
        self.goal_node = Node(goal[0], goal[1])  # node (x, y)
        self.space = space  # (min_x, max_x, min_y, max_y)
        self.obstalce_list = obstacle_list  # list of (x, y ,r)
        self.node_list = []

        # options
        self.max_iter = 5000
        self.goal_sample_rate = 0.1
        self.min_u = 1.0
        self.max_u = 3.0
        self.success_dist_thres = success_dist_thres
        self.collision_check_step = 0.2
        self.stepsize = 0.5

    def plan(self):
        self.node_list = [self.start_node]
        for i in range(self.max_iter):
            # Get random node
            rand_node = self.get_random_node()
            # Find neareast node
            nearest_node = self.find_nearest_node(self.node_list, rand_node)
            # Create new node
            u = self.stepsize*self.get_random_input(self.min_u, self.max_u)
            new_node = self.create_child_node(nearest_node, rand_node, u)
            # Collision check(Node, Path)
            node_collide = self.is_collide(new_node, self.obstalce_list)
            if node_collide:
                continue
            collide = self.is_path_collide(nearest_node, new_node, self.obstalce_list, self.collision_check_step)
            if collide:
                continue
            # Add to tree
            new_node.set_parent(nearest_node)
            self.node_list.append(new_node)
            # Goal check
            goal_reached = self.check_goal(
                new_node, self.success_dist_thres)
            if goal_reached:
                print(" [-] GOAL REACHED")
                return self.backtrace_path(new_node)
        return None

    @staticmethod
    def is_same_node(node1, node2):
        # Code
        return False

    def backtrace_path(self, node):
        current_node = node
        path = [current_node]
        reached_start_node = self.is_same_node(current_node, self.start_node)
        while not reached_start_node:
            current_node = current_node.parent
            path.append(current_node)
            reached_start_node = self.is_same_node(
                current_node, self.start_node)
        return path[::-1]

    def get_random_node(self):
        # Code
        # Recommendation : set random node as goal node occasionally
        if 1:
            return rand_node
        else:
            return self.goal_node

    def check_goal(self, node, success_dist_thres):
        # Code
        return False

    @staticmethod
    def create_child_node(nearest_node, rand_node, u):
        # Code
        new_node = Node(new_x, new_y)
        return new_node

    @staticmethod
    def get_random_input(min_u, max_u):
        # Code
        return np.random.uniform(min_u, max_u)

    @staticmethod
    def find_nearest_node(node_list, rand_node):
        # Code
        return node_list[min_index]

    @staticmethod
    def is_collide(node, obstacle_list):
        # Code
        return False

    @staticmethod
    def is_path_collide(node_from, node_to, obstacle_list, check_step=0.2):
        # Code
        return False


if __name__ == "__main__":
    start, goal, space, obstacle_list = map()

    success_dist_thres = 1.0
    rrt = RRT(start, goal, space, obstacle_list, success_dist_thres)
    path = rrt.plan()
    for node in path:
        print(" [-] x = %.2f, y = %.2f " % (node.x, node.y))

    # draw result
    _t = np.linspace(0, 2*np.pi, 30)
    for obs in obstacle_list:
        x, y, r = obs
        _x = x + r * np.cos(_t)
        _y = y + r * np.sin(_t)

        plt.plot(_x, _y, 'k-')

    goal_x = goal[0] + success_dist_thres * np.cos(_t)
    goal_y = goal[1] + success_dist_thres * np.sin(_t)
    plt.plot(goal_x, goal_y, 'g--')

    for i in range(len(path)-1):
        node_i = path[i]
        node_ip1 = path[i+1]
        plt.plot([node_i.x, node_ip1.x], [node_i.y, node_ip1.y], 'r.-')

    plt.axis("equal")
    plt.show()
