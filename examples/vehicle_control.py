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
dynamics = Dynamics.SymContinuous(state_dot, state, action)


#Construct cost to overtake
px1, py1, heading1, vel1, steer1 = state[:5]
px2, py2, heading2, vel2, steer2 = state[5:]
#cost for reference lane
L = 0.2*(py1 - 1.5)**2
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

#visualize the overtaking scenario
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
import numpy as np

def visualize(xs):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(0, 30)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect('equal')
    ax.axis("off")

    for boundary_y in [-3, 3]:
        ax.plot([0, 30], [boundary_y, boundary_y], 'k-', linewidth=1.0)

    for lane_y in [0, 1.5]:
        ax.plot([0, 30], [lane_y, lane_y], 'k--', linewidth=1.0)

    ego_length = 2.0
    ego_width = 1.0
    other_length = 2.0
    other_width = 1.0

    ego_rect = patches.Rectangle((0, 0), ego_length, ego_width, fc='r', ec='r', alpha=0.5)
    other_rect = patches.Rectangle((0, 0), other_length, other_width, fc='g', ec='g', alpha=0.5)
    ax.add_patch(ego_rect)
    ax.add_patch(other_rect)

    ego_trajectory, = ax.plot([], [], 'r-', label='Ego vehicle trajectory')
    other_trajectory, = ax.plot([], [], 'g-', label='Other vehicle trajectory')

    def init():
        ego_rect.set_xy((xs[0, 0] - ego_length / 2, xs[0, 1] - ego_width / 2))
        ego_rect.angle = np.degrees(xs[0, 2])
        other_rect.set_xy((xs[0, 5] - other_length / 2, xs[0, 6] - other_width / 2))
        other_rect.angle = np.degrees(xs[0, 7])
        ego_trajectory.set_data([], [])
        other_trajectory.set_data([], [])
        return ego_rect, other_rect, ego_trajectory, other_trajectory

    def update(frame):
        ego_center_x = xs[frame, 0]
        ego_center_y = xs[frame, 1]
        ego_angle = np.degrees(xs[frame, 2])

        other_center_x = xs[frame, 5]
        other_center_y = xs[frame, 6]
        other_angle = np.degrees(xs[frame, 7])

        ego_transform = Affine2D().rotate_deg_around(ego_center_x, ego_center_y, ego_angle) + ax.transData
        ego_rect.set_transform(ego_transform)
        ego_rect.set_xy((ego_center_x - ego_length / 2, ego_center_y - ego_width / 2))

        other_transform = Affine2D().rotate_deg_around(other_center_x, other_center_y, other_angle) + ax.transData
        other_rect.set_transform(other_transform)
        other_rect.set_xy((other_center_x - other_length / 2, other_center_y - other_width / 2))

        ego_trajectory.set_data(xs[:frame+1, 0], xs[:frame+1, 1])
        other_trajectory.set_data(xs[:frame+1, 5], xs[:frame+1, 6])
        return ego_rect, other_rect, ego_trajectory, other_trajectory

    ani = FuncAnimation(fig, update, frames=range(len(xs)), init_func=init, blit=True, interval=50)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Vehicle Overtaking Visualization with Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize(xs)
