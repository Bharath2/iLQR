import sympy as sp
import numpy as np
from ilqr import *

#state and action dimensions
n_x = 5
n_u = 2

#get symbolic variables
states, actions = GetSyms(n_x, n_u)
#Construct dynamics
px, py, heading, vel, steer = states
states_dot = sp.Matrix([
                vel*sp.cos(heading),
                vel*sp.sin(heading),
                vel*sp.tan(steer),
                actions[0],
                actions[1],])
dynamics = Dynamics.SymContinuous(states_dot, states, actions)

#Construct cost to follow circular path
terminal_cost = (sp.sqrt(px**2 + py**2 + 1e-6) - 2)**2 + (vel - 2)**2
running_cost  = terminal_cost + (actions[0]**2)*0.1 + (actions[1]**2)*0.1
cost = Cost.Symbolic(running_cost, terminal_cost, states, actions)

#initialise the controller
controller = iLQR(dynamics, cost)

#prediction Horizon
N = 50
#initial state
x0 = np.array([-3.0, 1.0, -0.2, 0.0, 0.0])
#initial guess
us_init = np.random.randn(N, n_u)*0.001
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init)
