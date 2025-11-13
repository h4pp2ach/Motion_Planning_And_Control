import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Long import VehicleModel_Long

class PID_Controller_ConstantSpace(object):
    def __init__(self, step_time, target_x, ego_x, constantSpace=0.0, P_Gain=0.0, D_Gain=0.0, I_Gain=0.0):
        self.space = constantSpace
        
        self.Kp = P_Gain
        self.Ki = I_Gain
        self.Kd = D_Gain
        self.dt = step_time
        
        self.error = target_x - ego_x - constantSpace
        self.error_prev = self.error
        self.error_sum = 0
        
        self.u = 0
    
    def ControllerInput(self, target_x, ego_x):
        self.error = target_x - ego_x - self.space
        self.error_sum += self.error * self.dt
        
        P_term = self.Kp * (self.error)
        I_term = self.Ki * (self.error_sum)
        D_term = self.Kd * (self.error - self.error_prev) / self.dt
        
        self.u = P_term + I_term + D_term
        self.error_prev = self.error

        

if __name__ == "__main__":
    
    step_time = 0.1
    simulation_time = 50.0
    m = 500.0
    
    vx_ego = []
    vx_target = []
    x_space = []
    time = []
    
    target_vehicle = VehicleModel_Long(step_time, m, 0.0, 30.0, 10.0)
    ego_vehicle = VehicleModel_Long(step_time, m, 0.5, 0.0, 10.0)
    controller = PID_Controller_ConstantSpace(step_time, target_vehicle.x, ego_vehicle.x,
                                              constantSpace = 10.0, P_Gain = 0.6, D_Gain = 1.3, I_Gain = 0.0)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        vx_ego.append(ego_vehicle.vx)
        vx_target.append(target_vehicle.vx)
        x_space.append(target_vehicle.x-ego_vehicle.x)
        controller.ControllerInput(target_vehicle.x, ego_vehicle.x)
        ego_vehicle.update(controller.u)
        target_vehicle.update(0.0)
        
        
    plt.figure(1)
    plt.plot(time, vx_ego,'r-',label = "ego_vx [m/s]")
    plt.plot(time, vx_target,'b-',label = "target_vx [m/s]")
    plt.xlabel('time [s]')
    plt.ylabel('Vx')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    
    plt.figure(2)
    plt.plot([0, time[-1]], [controller.space, controller.space], 'k-', label="reference")
    plt.plot(time, x_space,'b-',label = "space [m]")
    plt.xlabel('time [s]')
    plt.ylabel('x')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)    
    
    plt.show()

    