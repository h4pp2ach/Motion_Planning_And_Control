import numpy as np

class VehicleModel_Lat(object):
    def __init__(self, step_time, Vx, m=500, L=4, kv=0.005):
        self.dt = step_time
        self.m = m
        self.L = L
        self.kv = kv
        self.vx = Vx
        self.yawrate = 0
        self.Yaw = 0
        self.X = 0
        self.Y = 0

    def update(self, delta, Vx):
        self.vx = Vx
        self.delta = np.clip(delta,-0.5,0.5)
        self.yawrate = self.vx/(self.L+self.kv*self.vx**2)*self.delta
        self.Yaw = self.Yaw + self.dt*self.yawrate
        self.X = self.X + Vx*self.dt*np.cos(self.Yaw)
        self.Y = self.Y + Vx*self.dt*np.sin(self.Yaw)


        