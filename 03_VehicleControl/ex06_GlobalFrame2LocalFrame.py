import numpy as np
import matplotlib.pyplot as plt

class Global2Local(object):
    def __init__(self, num_points):
        self.n = num_points
        self.GlobalPoints = np.zeros((num_points,2))
        self.LocalPoints = np.zeros((num_points,2))
    
    def convert(self, points, Yaw_ego, X_ego, Y_ego):
        # Code
            
class PolynomialFitting(object):
    def __init__(self, num_degree, num_points):
        self.nd = num_degree
        self.np = num_points
        self.A = np.zeros((self.np, self.nd+1))
        self.b = np.zeros((self.np,1))
        self.coeff = np.zeros((num_degree+1,1))
        
    def fit(self, points):
        # Code

class PolynomialValue(object):
    def __init__(self, num_degree, num_points):
        self.nd = num_degree
        self.np = num_points
        self.x = np.zeros((1, self.nd+1))
        self.y = np.zeros((num_points, 1))
        self.points = np.zeros((self.np, 2))
        
    def calculate(self, coeff, x):
        # Code
        
        
if __name__ == "__main__":
    num_degree = 3
    num_point = 4
    points = np.array([[1,2],[3,3],[4,4],[5,5]])
    X_ego = 2.0
    Y_ego = 0.0
    Yaw_ego = np.pi/4
    x_local = np.arange(0.0, 10.0, 0.5)
    
    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    frameconverter.convert(points, Yaw_ego, X_ego, Y_ego)
    polynomialfit.fit(frameconverter.LocalPoints)
    polynomialvalue.calculate(polynomialfit.coeff, x_local)
    
    plt.figure(1)
    for i in range(num_point):
        plt.plot(points[i][0],points[i][1],'b.')
    plt.plot(X_ego,Y_ego,'ro',label = "Vehicle")
    plt.plot([X_ego, X_ego+0.2*np.cos(Yaw_ego)],[Y_ego, Y_ego+0.2*np.sin(Yaw_ego)],'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.title("Global Frame")
    plt.grid(True)    
    
    plt.figure(2)
    for i in range(num_point):
        plt.plot(frameconverter.LocalPoints[i][0],frameconverter.LocalPoints[i][1],'b.')
    plt.plot(polynomialvalue.points.T[0],polynomialvalue.points.T[1],'b:')
    plt.plot(0.0, 0.0,'ro',label = "Vehicle")
    plt.plot([0.0, 0.5],[0.0, 0.0],'r-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best")
    plt.axis((-10,10,-10,10))
    plt.title("Local Frame")
    plt.grid(True)   
    
    plt.show()