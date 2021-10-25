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

#get symbolic variables
state, action = GetSyms(n_x, n_u)
u = action[0]
sin, cos, omega = state

#Construct pendulum dynamics
m, g, l = 1, 10, 1
theta = sp.atan2(sin, cos)
#angular acceleration
alpha = (u - m*g*l*sp.sin(theta + np.pi))/(m*l**2)
theta_n = theta + omega*dt
state_n = sp.Matrix([sp.sin(theta_n),
                     sp.cos(theta_n),
                     omega + alpha*dt])
Pendulum = Dynamics.SymDiscrete(state_n, state, action)


#Construct cost to swing up Pendulum
#theta = 0 --> sin(theta) = 0, cos(theta) = 1
x_goal = np.array([0, 1, 0])
Q  = np.diag([0, 1, 0.1])
R  = np.diag([0.1])
QT = np.diag([0, 100, 100])
#Add constraints on torque input (2Nm to -2Nm)
cons = Constrain(u, max_u = [2], min_u = [-2])
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


#plot theta and action trajectory
import matplotlib.pyplot as plt
theta = np.arctan2(xs[:, 0], xs[:, 1])
theta = np.where(theta < 0, 2*np.pi+theta, theta)
plt.plot(theta)
plt.plot(us)
plt.show()
