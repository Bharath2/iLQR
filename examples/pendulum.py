'''
 swing up pendulum with limited torque
'''
import sympy as sp
import numpy as np

from ilqr import iLQR
from ilqr.utils import GetSyms, Constrain
from ilqr.containers import Dynamics, Cost

#state and action dimensions
n_x = 3
n_u = 1
#time step
dt = 0.025

#Construct pendulum dynamics
m, g, l = 1, 10, 1
def f(x, u):
    #current state
    sin, cos, omega = x
    theta = np.arctan2(sin, cos)
    #angular acceleration
    alpha = (u[0] - m*g*l*np.sin(theta + np.pi))/(m*l**2)
    #next theta
    theta_n = theta + omega*dt
    #return next state
    return np.array([np.sin(theta_n),
                     np.cos(theta_n),
                     omega + alpha*dt])
#call dynamics container
Pendulum = Dynamics.Discrete(f)


#Construct cost to swing up Pendulum
x, u = GetSyms(n_x, n_u)
#theta = 0 --> sin(theta) = 0, cos(theta) = 1
x_goal = np.array([0, 1, 0])
Q  = np.diag([0, 1, 0.1])
R  = np.diag([0.1])
QT = np.diag([0, 100, 100])
#Add constraints on torque input (2Nm to -2Nm)
cons = Bounded(u, high = [2], low = [-2])
SwingUpCost = Cost.QR(Q, R, QT, x_goal, cons)


#initialise the controller
controller = iLQR(Pendulum, SwingUpCost)

#initial state
#theta = pi --> sin(theta) = 0, cos(theta) = -1
x0 = np.array([0, -1, 0])
#initial guess
us_init = np.random.randn(200, n_u)*0.01
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init)


#Plot theta and action trajectory
import matplotlib.pyplot as plt
theta = np.arctan2(xs[:, 0], xs[:, 1])
theta = np.where(theta < 0, 2*np.pi+theta, theta)
plt.plot(theta)
plt.plot(us)
plt.show()
