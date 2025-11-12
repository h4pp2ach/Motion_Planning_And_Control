import matplotlib.pyplot as plt
import numpy as np
def map():
    start = [0, 0]
    goal = [6, 10]
    space = [-2, 15, -2, 15]  # min_x, max_x, min_y, max_y
    obstacle_list = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                        (9, 5, 2), (8, 10, 1)]  # (x, y, radius)
    return start, goal, space, obstacle_list

if __name__ == "__main__":
    start, goal, space, obstacle_list = map()
    plt.figure(figsize=(8,8))
    plt.plot(start[0], start[1], 'bs',  markersize=7)
    plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
    plt.plot(goal[0], goal[1], 'rs',  markersize=7)
    plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
    _t = np.linspace(0, 2*np.pi, 30)
    for obs in obstacle_list:
        x, y, r = obs
        _x = x + r * np.cos(_t)
        _y = y + r * np.sin(_t)
        plt.plot(_x, _y, 'k-')
    plt.grid(True)
    plt.show()