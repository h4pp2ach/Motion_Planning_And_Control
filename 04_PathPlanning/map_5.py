import matplotlib.pyplot as plt
import numpy as np
class Obstacle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def plot(self, color='k'):
        theta = np.linspace(0, np.pi*2, num=30)
        x = self.x + self.r * np.cos(theta)
        y = self.y + self.r * np.sin(theta)
        plt.plot(x, y, color=color)

    def is_inside(self, x, y):
        dist = np.hypot(x - self.x, y - self.y) # Distance 
        if dist <= self.r:
            return True
        else:
            return False
def map():
    #np.random.seed(2)
    min_x, max_x = -20, 20
    min_y, max_y = -20, 20

    space = [min_x, max_x, min_y, max_y]

    start = np.random.uniform(low=-20, high=-5, size=2)
    goal = np.random.uniform(low=5, high=20, size=2)

    obstacles = []
    for i in range(25):
        x = np.random.uniform(low=min_x, high=max_x, size=1)
        y = np.random.uniform(low=min_y, high=max_y, size=1)
        r = np.random.uniform(low=1.0, high=5.0)
        obstacle = Obstacle(x, y, r)

        if not obstacle.is_inside(start[0], start[1]) and not obstacle.is_inside(goal[0], goal[1]):
            obstacles.append(obstacle)
    return start, goal, space, obstacles
    
if __name__ == "__main__":
    start, goal, space, obstacles = map(-20, 20, -20, 20)
    plt.figure(figsize=(8,8))
    plt.plot(start[0], start[1], 'bs',  markersize=7)
    plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
    plt.plot(goal[0], goal[1], 'rs',  markersize=7)
    plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
    _t = np.linspace(0, 2*np.pi, 30)
    for obs in obstacles:
        obs.plot()
    plt.grid(True)
    plt.show()