import numpy as np
import matplotlib.pyplot as plt

def lane(X_ref, lanewidth=4.0):
    Y_ref_L = 2.0-2*np.cos(X_ref/10)+lanewidth/2
    Y_ref_R = 2.0-2*np.cos(X_ref/10)-lanewidth/2
    return Y_ref_L, Y_ref_R

if __name__ == "__main__":
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref_L, Y_ref_R = lane(X_ref)
    plt.figure(figsize=(13,2))
    plt.plot(X_ref, Y_ref_L, 'k--')
    plt.plot(X_ref, Y_ref_R, 'k--')
    plt.grid(True)
    plt.show()