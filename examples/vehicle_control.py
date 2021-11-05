'''
 Vehicle Overtaking
 Adjust cost and initial state to get desired behaviors
'''

import sympy as sp
import numpy as np
from ilqr import *

def vehicle_kinematics(state, action):
    px, py, heading, vel, steer = state
    accel, steer_vel = action

    state_dot = sp.Matrix([
                    vel*sp.cos(heading),
                    vel*sp.sin(heading),
                    vel*sp.tan(steer),
                    accel,
                    steer_vel])

    return state_dot


#state and action dimensions
n_x = 10
n_u = 2

#get symbolic variables
state, action = GetSyms(n_x, n_u)

#Construct dynamics
state_dot = sp.Matrix([0.0]*n_x)
# ego vehicle kinematics
state_dot[:5, :] = vehicle_kinematics(state[:5], action)
# other vehicle kinematics (constant velocity and steering)
state_dot[5:, :] = vehicle_kinematics(state[5:], [0, 0])
#construct
dynamics = Dynamics.SymContinuous(state_dot, state, actio)


#Construct cost to overtake
px1, py2, heading1, vel1, steer1 = state[:5]
px2, py2, heading2, vel2, steer2 = state[5:]
#cost for reference lane
L = 0.5*(py1 - 1.5)**2
#cost on velocity
L += (vel1*sp.cos(heading1) - 2)**2 + (vel1 - 2)**2
#penality on actions
L += 0.1*action[1]**2 + 0.1*action[0]**2

#collision avoidance (do not cross ellipse around the vehicle)
L += SoftConstrain([((px1 - px2)/4.5)**2 + ((py1 - py2)/2)**2 - 1])
#constrain steering angle and y-position
L += Bounded([py1, steer1], high = [2.5, 0.523], low = [-2.5, -0.523])
#construct
cost = Cost.Symbolic(L, 0, state, action)

#initialise the controller
controller = iLQR(dynamics, cost)
#prediction Horizon
N = 200
#initial state
x0 = np.array([0, 1.5, 0, 1, 0,
               4, 1.5, 0, 1, 0])
#initil guess
us_init = np.random.randn(N, n_u)*0.0001
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init, 100)
