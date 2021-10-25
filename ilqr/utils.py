import sympy as sp
import numpy as np
from numba import njit


def GetSyms(n_x, n_u):
  '''
      Returns matrices with symbolic variables for states and actions
      n_x: state size
      n_u: action size
  '''

  x = sp.IndexedBase('x')
  u = sp.IndexedBase('u')
  xs = sp.Matrix([x[i] for i in range(n_x)])
  us = sp.Matrix([u[i] for i in range(n_u)])
  return xs, us


def Constrain(u, max_u, min_u):
    '''
    Logarithmic barrier function to constrain variables
    '''
    cost = 0
    for i in range(u.shape[0]):
        #normalise u[i] to vary from -1 to 1
        diff = (max_u[i] - min_u[i])/2
        u_norm = (u[i] - min_u[i])/diff - 1
        #constrain u_norm from -1 to 1
        cost -= sp.log(1 - u_norm + 1e-4)
        cost -= sp.log(1 + u_norm + 1e-4)
    return 0.1*cost


def Smooth_abs(x, alpha = 0.25):
    '''
    smooth absolute value
    '''
    return sp.sqrt(x**2 + alpha**2) - alpha


@njit
def FiniteDiff(fun, x, u, i, eps):
  '''
     Finite difference approximation
  '''

  args = (x, u)
  fun0 = fun(x, u)

  m = x.size
  n = args[i].size

  Jac = np.zeros((m, n))
  for k in range(n):
    args[i][k] += eps
    Jac[:, k] = (fun(args[0], args[1]) - fun0)/eps
    args[i][k] -= eps

  return Jac



def sympy_to_numba(f, args, redu = True):
    '''
       Converts sympy matrix or expression to numba jitted function
    '''
    if isinstance(f, sp.Matrix):
        #To convert all elements to floats
        m, n = f.shape
        f += 1e-64*np.ones((m, n))

        #To eleminate extra dimension
        if (n == 1 or m == 1) and redu:
            if n == 1: f = f.T
            f = sp.Array(f)[0, :]
            f = njit(sp.lambdify(args, f, modules = [{'atan2':np.arctan2}, 'numpy']))
            f_new = lambda *args: np.array(f(*args))
            return njit(f_new)

    f = sp.lambdify(args, f, modules = [{'atan2':np.arctan2}, 'numpy'])
    return njit(f)
