import sympy as sp
import numpy as np

from ilqr import iLQR
from ilqr.utils import GetSyms, Constrain
from ilqr.containers import Dynamics, Cost
from quaternion import *

#params
dt = 0.01
mass = 0.18 # kg
g = 9.81 # m/s^2
I = np.array([(0.00025, 0, 2.55e-6),
              (0, 0.000232, 0),
              (2.55e-6, 0, 0.0003738)]);
invI = np.linalg.inv(I)

#Quadrotor dynamics
def f(state, actions):
    F, M1, M2, M3 = actions
    x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
    quat = np.array([qw,qx,qy,qz])

    bRw = Quaternion(quat).as_rotation_matrix() # world to body rotation matrix
    wRb = bRw.T # orthogonal matrix inverse = transpose
    # acceleration - Newton's second law of motion
    accel = 1.0 / mass*(wRb.dot(np.array([[0, 0, F]]).T)
                - np.array([[0, 0, mass*g]]).T)
    # angular velocity - using quternion
    # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
    K_quat = 2.0; # this enforces the magnitude 1 constraint for the quaternion
    quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
    qdot = (-1.0/2) * np.array([[0, -p, -q, -r],
                                [p,  0, -r,  q],
                                [q,  r,  0, -p],
                                [r, -q,  p,  0]]).dot(quat) + K_quat * quaterror * quat;
    # angular acceleration - Euler's equation of motion
    # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
    omega = np.array([p, q, r])
    pqrdot = invI.dot( np.array([M1, M2, M3]) - np.cross(omega, I.dot(omega)) )
    state_dot = np.r_([[xdot, ydot, zdot],
                      accel, qdot, pqrdot]
    return state_dot


Quadrotor = Dynamics.Continuous(f, dt)
