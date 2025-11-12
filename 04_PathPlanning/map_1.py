import matplotlib.pyplot as plt
def map():
    start = (5, 5)
    goal = (55, 5)

    ox, oy = [], []

    for i in range(61):
        ox.append(i)
        oy.append(0)
    for i in range(61):
        ox.append(0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60)
    for i in range(61):
        ox.append(60)
        oy.append(i)
    for i in range(51):
        ox.append(30)
        oy.append(i)
    for i in range(21):
        ox.append(i)
        oy.append(20)
        ox.append(40+i)
        oy.append(20)
    for i in range(41):
        ox.append(10+i)
        oy.append(35)
    obstacle_map = [ox, oy]    
    return start, goal, obstacle_map

if __name__ == "__main__":
    start, goal, omap = map()
    plt.figure(figsize=(8,8))
    plt.plot(start[0], start[1], 'bs',  markersize=7)
    plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
    plt.plot(goal[0], goal[1], 'rs',  markersize=7)
    plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
    plt.plot(omap[0], omap[1], '.k',  markersize=10)
    plt.grid(True)
    plt.show()