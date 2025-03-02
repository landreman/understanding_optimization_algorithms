import numpy as np
from scipy.optimize import least_squares
from understanding_optimization_algorithms import trf_no_bounds


def rosenbrock_residuals(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

def rosenbrock_jacobian(x):
    return np.array([[-20 * x[0], 10], [-1, 0]])

def test_scipy_trf():
    x0 = np.array([0, 0])
    res = least_squares(rosenbrock_residuals, x0, jac=rosenbrock_jacobian, method="trf", verbose=2)
    np.testing.assert_allclose(res.x, [1, 1], rtol=1e-14)
    assert res.success


def test_trf():
    x0 = np.array([0, 0])
    # Order of arguments:
    # fun, jac, x0, ftol, xtol, gtol, max_nfev,
    # x_scale, loss_function, tr_solver, tr_options, verbose,
    res = trf_no_bounds(
        rosenbrock_residuals,
        rosenbrock_jacobian,
        x0,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100,
        x_scale=1.0,
        loss_function=None,
        tr_solver="exact",
        tr_options=None,
        verbose=2,
    )
    np.testing.assert_allclose(res.x, [1, 1], rtol=1e-14)
    assert res.success
