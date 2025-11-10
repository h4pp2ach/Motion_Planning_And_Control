import numpy as np

class VehicleModel_Long(object):
    def __init__(self, step_time, m, Ca, x_init, Vx_init):
        self.x = x_init
        self.vx = Vx_init
        self.ax = 0.0
        self.yawRate = 0.0
        self.dt = step_time
        self.theta = 0.0
        self.m = m
        self.g = 9.81
        self.C = Ca/m
        self.delta = 0.0
        
    def update(self, a_x):
        self.x = self.x + self.dt*self.vx + (self.dt**2)*self.ax/2
        self.vx = self.vx + self.dt*(self.ax)
        self.ax = np.clip(a_x - self.C*(self.vx**2)-self.g*np.sin(self.theta), -2.0, 2.0)

        