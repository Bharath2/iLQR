'''
 Quadrotor: fly from hover at the origin to hover at a goal position
 Dynamics and parameters are the same as in Quadrotor-Simulation (Quadrotor/quadrotor.py)
   state  = [x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r]
   action = thrusts of the four rotors [F1, F2, F3, F4] (N)
'''

import sympy as sp
import numpy as np

from ilqr import iLQR
from ilqr.utils import GetSyms, Bounded
from ilqr.containers import Dynamics, Cost

#state and action dimensions
n_x = 13
n_u = 4
#time step
dt = 0.01

#params
mass = 0.18 # kg
g = 9.81 # m/s^2
arm_length = 0.086 # m
I = np.array([(0.00025, 0, 2.55e-6),
              (0, 0.000232, 0),
              (2.55e-6, 0, 0.0003738)])
invI = np.linalg.inv(I)
#limits on each rotor's thrust
minF = 0.0
maxF = 2.0*mass*g
#rotor thrusts to total thrust and moments
#  [ F  ]         [ F1 ]
#  | M1 |  = A *  | F2 |
#  | M2 |         | F3 |
#  [ M3 ]         [ F4 ]
l  = arm_length
km = 1.5e-9
kf = 6.11e-8
k  = km/kf
A  = np.array([[ 1,  1,  1,  1],
               [ 0,  l,  0, -l],
               [-l,  0,  l,  0],
               [ k, -k,  k, -k]])

#Quadrotor dynamics
def f(state, rotor_thrusts):
    F, M1, M2, M3 = A @ rotor_thrusts
    x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
    quat = np.array([qw, qx, qy, qz])

    # body z axis in the world frame: wRb @ [0, 0, 1] = third row of bRw,
    # the world to body rotation matrix of the (normalized) quaternion
    nw, nx, ny, nz = quat/np.sqrt(quat @ quat)
    zB = np.array([2*(nx*nz - nw*ny),
                   2*(ny*nz + nw*nx),
                   1 - 2*(nx**2 + ny**2)])
    # acceleration - Newton's second law of motion
    accel = F/mass*zB - np.array([0.0, 0.0, g])
    # angular velocity - using quternion
    # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
    K_quat = 2.0 # this enforces the magnitude 1 constraint for the quaternion
    quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
    qdot = (-1.0/2) * np.array([[0.0, -p, -q, -r],
                                [p,  0.0, -r,  q],
                                [q,  r,  0.0, -p],
                                [r, -q,  p,  0.0]]) @ quat + K_quat*quaterror*quat
    # angular acceleration - Euler's equation of motion
    # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
    omega = np.array([p, q, r])
    pqrdot = invI @ (np.array([M1, M2, M3]) - np.cross(omega, I @ omega))
    state_dot = np.concatenate((np.array([xdot, ydot, zdot]), accel, qdot, pqrdot))
    return state_dot

#call dynamics container
Quadrotor = Dynamics.Continuous(f, dt)


#Construct cost to fly to the goal and hover there
x, u = GetSyms(n_x, n_u)
#hover at (1, 1, 1) with level attitude (qw = 1)
x_goal  = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
#thrust of each rotor needed to hover
u_hover = mass*g/4*np.ones(n_u)
#weights:   position       velocity        quaternion          angular velocity
Q  = np.diag([1, 1, 1,     0.1, 0.1, 0.1,  1, 1, 1, 1,         0.01, 0.01, 0.01])
R  = np.diag([1, 1, 1, 1])
QT = np.diag([100, 100, 100, 10, 10, 10,   10, 10, 10, 10,     1, 1, 1])
#penalise deviation from the goal and from hover thrust
er = x - sp.Matrix(x_goal)
du = u - sp.Matrix(u_hover)
L  = (er.T@Q@er + du.T@R@du)[0]
Lf = (er.T@QT@er)[0]
#keep each rotor's thrust within its limits
L += Bounded(u, high = [maxF/4]*n_u, low = [minF/4]*n_u)
FlyToGoalCost = Cost.Symbolic(L, Lf, x, u)


#initialise the controller
controller = iLQR(Quadrotor, FlyToGoalCost)

#Prediction Horizon (2 seconds)
N = 200
#initial state: hover at the origin
x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype = float)
#initial guess: hover thrust (inside the thrust limits)
us_init = u_hover + np.random.randn(N, n_u)*0.001
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init, 100)


#Animate the flight with position and rotor thrust trajectories
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def body_to_world(quat):
    '''rotation matrix wRb (transpose of bRw used in the dynamics)'''
    w, x, y, z = quat/np.linalg.norm(quat)
    bRw = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]])
    return bRw.T

def visualize(xs, us):
    t = np.arange(len(xs))*dt
    fig = plt.figure(figsize = (11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios = [1.2, 1])
    ax = fig.add_subplot(gs[:, 0], projection = '3d')
    ax_p = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex = ax_p)

    #3D view: planned path, start, goal
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-0.2, 1.2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.plot(*xs[:, :3].T, ':', color = 'gray', linewidth = 1)
    ax.scatter(*x0[:3], color = 'k', s = 20)
    ax.scatter(*x_goal[:3], color = 'tab:green', marker = '*', s = 120)
    arm13, = ax.plot([], [], [], '-', color = 'tab:red', linewidth = 3)
    arm24, = ax.plot([], [], [], '-', color = 'tab:blue', linewidth = 3)
    thrust, = ax.plot([], [], [], '-', color = 'tab:orange', linewidth = 2)
    trail, = ax.plot([], [], [], '-', color = 'tab:green', linewidth = 1.5)
    title = fig.suptitle('quadrotor: hover to hover   t = 0.00 s')

    #rotors in the body frame (drawn twice the real size)
    rotors = 2*arm_length*np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])

    #position and rotor thrusts
    for i, (c, ls) in enumerate(zip(['tab:red', 'tab:blue', 'tab:green'], ['-', '--', '-'])):
        ax_p.plot(t, xs[:, i], color = c, linestyle = ls, label = 'xyz'[i])
        ax_p.axhline(x_goal[i], color = c, linestyle = '--', linewidth = 0.8)
    ax_p.set_ylabel('position (m)')
    ax_p.legend(loc = 'lower right')
    ax_u.plot(t[:-1], us)
    ax_u.axhline(maxF/4, color = 'k', linestyle = '--', linewidth = 0.8)
    ax_u.axhline(minF/4, color = 'k', linestyle = '--', linewidth = 0.8)
    ax_u.legend(['F1', 'F2', 'F3', 'F4'], loc = 'upper right', ncol = 4, fontsize = 8)
    ax_u.set_ylabel('rotor thrust (N)')
    ax_u.set_xlabel('time (s)')
    cursors = [ax_p.axvline(0, color = 'gray'), ax_u.axvline(0, color = 'gray')]
    fig.tight_layout()

    def update(i):
        pos, wRb = xs[i, :3], body_to_world(xs[i, 6:10])
        r = pos + rotors@wRb.T
        arm13.set_data_3d(r[[0, 2], 0], r[[0, 2], 1], r[[0, 2], 2])
        arm24.set_data_3d(r[[1, 3], 0], r[[1, 3], 1], r[[1, 3], 2])
        #thrust along the body z axis, 0.1 m per N
        F = us[min(i, len(us) - 1)].sum()
        tip = pos + 0.1*F*wRb[:, 2]
        thrust.set_data_3d([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]])
        trail.set_data_3d(xs[:i+1, 0], xs[:i+1, 1], xs[:i+1, 2])
        title.set_text(f'quadrotor: hover to hover   t = {t[i]:.2f} s')
        for cursor in cursors:
            cursor.set_xdata([t[i], t[i]])
        return arm13, arm24, thrust, trail, title, *cursors

    #every other step at half speed, then hold the last frame
    frames = list(range(0, len(xs), 2)) + [len(xs) - 1]*25
    ani = FuncAnimation(fig, update, frames = frames, interval = 4000*dt)
    plt.show()

visualize(xs, us)
