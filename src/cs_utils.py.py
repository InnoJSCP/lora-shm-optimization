import numpy as np
from scipy.fft import dct
import cvxpy as cp

def generate_sensing_matrix(M, N, seed=42):
    np.random.seed(seed)
    return np.random.randn(M, N) / np.sqrt(M)

def compress(x, Phi):
    return Phi @ x

def recover_bp(y, Phi, epsilon=1e-3, Psi=None):
    """
    Basis Pursuit recovery with DCT sparsity basis.
    """
    N = Phi.shape[1]
    if Psi is None:
        # DCT basis: x = Psi @ alpha, alpha = DCT(x)
        Psi = dct(np.eye(N), norm='ortho')
    alpha = cp.Variable(N)
    objective = cp.Minimize(cp.norm(alpha, 1))
    constraints = [cp.norm(y - Phi @ Psi @ alpha, 2) <= epsilon]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return Psi @ alpha.value

def nrmse(x, x_hat):
    return np.sqrt(np.mean((x - x_hat)**2)) / np.max(np.abs(x))

def quality_metric(x, x_hat):
    return 1.0 / (1.0 + nrmse(x, x_hat))