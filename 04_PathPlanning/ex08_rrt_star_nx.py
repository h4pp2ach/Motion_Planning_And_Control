import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from map_5 import map

class RRTStar(object):
    def __init__(self, start, goal, config):
        self.G = nx.DiGraph() # Directed Graph
        self.G.add_nodes_from([(-1, {'cost': 0, 'x': start[0], 'y': start[1]})]) # Add Start node [(Node ID, {'Cost', 'x', 'y'})]
        self.start = start
        self.goal = goal
        self.config = config

    # Random point generation
    def sample_free(self, obstacles, space):
        min_x, max_x, min_y, max_y = space
        # Code
        return np.array([rand_x, rand_y])

    # Search nearest node
    def get_nearest(self, rand_node):
        # Code
        return nearest_node_id

    # Node connection
    def steer(self, node_from, node_to, u=None):
        # Code
        return new_x, new_y

    # Returns node(2d-array with position info.) corresponding to the node id
    def get_node(self, node_id):
        node = np.array([self.G.nodes[node_id]['x'], self.G.nodes[node_id]['y']])
        return node

    # Collision Check
    def is_collision_free(self, node_from, node_to, obstacles, step=0.2):
        # Code
        # Collision check step size : u
        # Collision check : path(connection) between 2 nodess
        return False

    # Find adjacent nodes
    def get_near_node_ids(self, new_node, draw):
        # Code
        # Recommendation : Searching distance proportional to log(# of nodes in tree)/(# of nodes in tree)
        return near_node_ids

    # Add node to tree
    def add_node(self, node_id, x, y):
        self.G.add_node(node_id, x=x, y=y)

    # Get cost of designated node
    def get_node_cost(self, node_id):
        return self.G.nodes[node_id]['cost']

    # Calculate the distance between 2 nodes
    def get_distance(self, node_from_id, node_to_id):
        # Code
        return dist

    # Add edge(Connection) between 2 nodes
    def add_edge(self, node_from_id, node_to_id):
        self.G.add_edge(node_from_id, node_to_id)

    # Set cost to the node
    def set_node_cost(self, node_id, cost):
        self.G.nodes[node_id]['cost'] = cost

    # Get parent node of designated node : predecessors
    def get_parent(self, node_id):
        parents = list(self.G.predecessors(node_id))
        if len(parents) > 0:
            return parents[0]
        else:
            return None

    # Delete connections between 2 nodes
    def remove_edge(self, node_from_id, node_to_id):
        self.G.remove_edge(node_from_id, node_to_id)

    # Check goal
    def check_goal_by_id(self, node_id):
        node = self.G.nodes[node_id]
        dx = node['x'] - self.goal[0]
        dy = node['y'] - self.goal[1]
        dist = np.hypot(dx, dy)
        # Regard as destination, if distance between the node and the goal is smaller than threshold.
        if dist < self.config["goal_range"]:
            return True
        else:
            return False


if __name__ == '__main__':

    start, goal, space, obstacles = map()
    for obs in obstacles:
        obs.plot()

    config = {
        "eta": 3.0,
        "gamma_rrt_star": 4.0,
        "goal_sample_rate": 0.05,
        "min_u": 1.0,
        "max_u": 3.0,
        "goal_range" : 1.0
    }

    rrt_star = RRTStar(start, goal, config)

    is_first_node = True
    goal_node_id = None

    for i in range(1000):
    ## Create Random Node 
        rand_node = rrt_star.sample_free(obstacles, space)
        # plt.plot(rand_node[0], rand_node[1], '.')
    ## Find Nearest Node
        nearest_node_id = rrt_star.get_nearest(rand_node)
        nearest_node = rrt_star.get_node(nearest_node_id)
    ## Connect new node to the nearest node
        new_node = rrt_star.steer(nearest_node, rand_node)
        # plt.plot(new_node[0], new_node[1], 's')
    ## Check Collision of the new node
        if rrt_star.is_collision_free(nearest_node, new_node, obstacles):
    ## Find adjacent nodes
            near_node_ids = rrt_star.get_near_node_ids(new_node, draw=True)
            rrt_star.add_node(i, new_node[0], new_node[1])
            if is_first_node:
                rrt_star.add_edge(-1, i)
                is_first_node = False
            plt.plot(new_node[0], new_node[1], 's')

            min_node_id = nearest_node_id
            min_cost = rrt_star.get_node_cost(nearest_node_id) + rrt_star.get_distance(i, nearest_node_id)
    ## Find a node with minimum cost among adjacent nodes
            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(near_node, new_node, obstacles):
                    cost = rrt_star.get_node_cost(near_node_id) + rrt_star.get_distance(near_node_id, i)
                    if cost < min_cost:
                        min_node_id = near_node_id
                        min_cost = cost

            rrt_star.set_node_cost(i, min_cost)
            rrt_star.add_edge(min_node_id, i)

    ## Rewire the tree with adjacent nodes
            for near_node_id in near_node_ids:
                near_node = rrt_star.get_node(near_node_id)
                if rrt_star.is_collision_free(new_node, near_node, obstacles):
                    cost = rrt_star.get_node_cost(i) + rrt_star.get_distance(i, near_node_id)
                    if cost < rrt_star.get_node_cost(near_node_id):
                        parent_node_id = rrt_star.get_parent(near_node_id)
                        if parent_node_id is not None:
                            rrt_star.remove_edge(parent_node_id, near_node_id)
                            rrt_star.add_edge(i, near_node_id)
    ## Check Goal
            if rrt_star.check_goal_by_id(i):
                goal_node_id = i
                break
    ## Plotting
    for e in rrt_star.G.edges:
        v_from = rrt_star.G.nodes[e[0]]
        v_to = rrt_star.G.nodes[e[1]]

        plt.plot([v_from['x'], v_to['x']], [v_from['y'], v_to['y']], 'b-')
        plt.text(v_to['x'], v_to['y'], e[1])

    if goal_node_id is not None:
        path = nx.shortest_path(rrt_star.G, source=-1, target=goal_node_id)
        xs = []
        ys = []

        for node_id in path:
            node = rrt_star.G.nodes[node_id]
            xs.append(node['x'])
            ys.append(node['y'])
        plt.plot(xs, ys, 'r-', lw=3)



    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bx')

    plt.axis("equal")
    plt.show()
