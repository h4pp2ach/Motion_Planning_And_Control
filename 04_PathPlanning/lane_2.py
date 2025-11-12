import numpy as np
import matplotlib.pyplot as plt

def lane(X_ref, lanewidth=4.0):
    Y_ref_L = []
    Y_ref_R = []
    isLaneValid_L = []
    isLaneValid_R = []
    for i in range(len(X_ref)):
        if (X_ref[i] > 20.0 and X_ref[i] < 40.0):
            Y_ref_L.append(0.0)
            Y_ref_R.append(2.0-2*np.cos(X_ref[i]/10)-lanewidth/2)
            isLaneValid_L.append(False)
            isLaneValid_R.append(True)
        elif (X_ref[i] > 60.0 and X_ref[i] <80.0):
            Y_ref_L.append(2.0-2*np.cos(X_ref[i]/10)+lanewidth/2)
            Y_ref_R.append(0.0)
            isLaneValid_L.append(True)
            isLaneValid_R.append(False)
        else:
            Y_ref_L.append(2.0-2*np.cos(X_ref[i]/10)+lanewidth/2)
            Y_ref_R.append(2.0-2*np.cos(X_ref[i]/10)-lanewidth/2)
            isLaneValid_L.append(True)
            isLaneValid_R.append(True)
    return Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R

if __name__ == "__main__":
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref_L, Y_ref_R, isLaneValid_L, isLaneValid_R = lane(X_ref)
    plt.figure(figsize=(13,2))
    plt.plot(X_ref, Y_ref_L, 'k--')
    plt.plot(X_ref, Y_ref_R, 'k--')
    plt.grid(True)
    plt.show()