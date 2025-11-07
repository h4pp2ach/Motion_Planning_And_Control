import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average = 2):
        self.y_estimate = y_initial_measure
        self.num_average = num_average
        self.dataQueue = deque([self.y_estimate])
        
    def estimate(self, y_measure):
        
        self.dataQueue.append(y_measure)
        numOfData = len(self.dataQueue)
        
        if (numOfData > self.num_average):
            self.dataQueue.popleft() # Queue 안의 원소의 갯수는 Window Size로 유지
            numOfData -= 1
            
            self.y_estimate += ( y_measure-self.dataQueue[0] ) / self.num_average
        
        else:
            self.y_estimate = (sum(self.dataQueue)) / numOfData

    
if __name__ == "__main__":
    #signal = pd.read_csv("./Data/example_Filter_2.csv")     
    signal = pd.read_csv("./Data/example_Filter_3.csv")

    error = 0
    y_estimate = MovingAverageFilter(signal.y_measure[0], 12)
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i])
        signal.y_estimate[i] = y_estimate.y_estimate
        error += (signal.y_true[i] - signal.y_estimate[i])**2

    print(error)
    
    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



