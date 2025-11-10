from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.2, D_Gain=0.7, I_Gain=0.02):
        self.Kp = P_Gain
        self.Ki = I_Gain
        self.Kd = D_Gain
        self.dt = step_time
        
        self.error = reference - measure
        self.error_prev = self.error
        self.error_sum = 0.0
        
        self.u = 0.0
    
    def ControllerInput(self, reference, measure):
        self.error = reference - measure
        self.error_sum += self.error * self.dt
        
        P_term = self.Kp * (self.error)
        I_term = self.Ki * (self.error_sum)
        D_term = self.Kd * (self.error - self.error_prev) / self.dt
        
        self.u = P_term + I_term + D_term
        self.error_prev = self.error
        
        
if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.0, 0.99, 0.05)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        controller.ControllerInput(target_y, plant.y_measure[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r-',label = "Vehicle Position")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
