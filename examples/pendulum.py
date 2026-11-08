'''
 swing up pendulum with limited torque
'''
import sympy as sp
import numpy as np

from ilqr import iLQR
from ilqr.utils import GetSyms, Constrain, Bounded
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
#Prediction Horizon (7.5 seconds: time to swing up and then balance)
N = 300
#initial guess
us_init = np.random.randn(N, n_u)*0.01
#get optimal states and actions (the longer horizon needs more iterations)
xs, us, cost_trace = controller.fit(x0, us_init, 200)


#Animate the swing up with angle and torque trajectories
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize(xs, us):
    theta = np.arctan2(xs[:, 0], xs[:, 1])
    theta = np.where(theta < 0, 2*np.pi + theta, theta)
    t = np.arange(len(xs))*dt

    fig = plt.figure(figsize = (10, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios = [1, 1.3])
    ax = fig.add_subplot(gs[:, 0])
    ax_th = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex = ax_th)

    #pendulum (theta = 0 is upright)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot(0, 0, 'ko', markersize = 6)
    rod, = ax.plot([], [], 'k-', linewidth = 3)
    bob, = ax.plot([], [], 'o', color = 'tab:red', markersize = 18)
    title = fig.suptitle('pendulum swing up   t = 0.00 s')

    #angle and torque
    ax_th.plot(t, np.degrees(theta))
    for upright in [0, 360]:
        ax_th.axhline(upright, color = 'k', linestyle = '--', linewidth = 0.8)
    ax_th.set_ylabel('angle (deg)')
    ax_u.plot(t[:-1], us[:, 0], color = 'tab:orange')
    for limit in [-2, 2]:
        ax_u.axhline(limit, color = 'k', linestyle = '--', linewidth = 0.8)
    ax_u.set_ylabel('torque (Nm)')
    ax_u.set_xlabel('time (s)')
    cursors = [ax_th.axvline(0, color = 'gray'), ax_u.axvline(0, color = 'gray')]
    fig.tight_layout()

    def update(i):
        x, y = l*np.sin(theta[i]), l*np.cos(theta[i])
        rod.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        title.set_text(f'pendulum swing up   t = {t[i]:.2f} s')
        for cursor in cursors:
            cursor.set_xdata([t[i], t[i]])
        return rod, bob, title, *cursors

    #every other step in real time, then hold the last frame
    frames = list(range(0, len(xs), 2)) + [len(xs) - 1]*20
    ani = FuncAnimation(fig, update, frames = frames, interval = 2000*dt)
    plt.show()

visualize(xs, us)
