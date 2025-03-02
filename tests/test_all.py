import numpy as np
from scipy.optimize import least_squares
from understanding_optimization_algorithms import trf_no_bounds


def rosenbrock_residuals(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def rosenbrock_jacobian(x):
    return np.array([[-20 * x[0], 10], [-1, 0]])


# Copied from https://github.com/scipy/scipy/blob/main/scipy/optimize/tests/test_least_squares.py
class ExponentialFittingProblem:
    """Provide data and function for exponential fitting in the form
    y = a + exp(b * x) + noise."""

    def __init__(
        self, a, b, noise, n_outliers=1, x_range=(-1, 1), n_points=11, random_seed=42
    ):
        rng = np.random.RandomState(random_seed)
        self.m = n_points
        self.n = 2

        self.p0 = np.zeros(2)
        self.x = np.linspace(x_range[0], x_range[1], n_points)

        self.y = a + np.exp(b * self.x)
        self.y += noise * rng.randn(self.m)

        outliers = rng.randint(0, self.m, n_outliers)
        self.y[outliers] += 50 * noise * rng.rand(n_outliers)

        self.p_opt = np.array([a, b])

    def fun(self, p):
        return p[0] + np.exp(p[1] * self.x) - self.y

    def jac(self, p):
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = self.x * np.exp(p[1] * self.x)
        return J


def test_scipy():
    for method in ["trf", "lm", "dogbox"]:
        x0 = np.array([0, 0])
        res = least_squares(
            rosenbrock_residuals, x0, jac=rosenbrock_jacobian, method=method, verbose=2
        )
        np.testing.assert_allclose(res.x, [1, 1], rtol=1e-14)
        assert res.success

        prob = ExponentialFittingProblem(a=3, b=2, noise=0.0)
        x0 = np.zeros(2)
        res = least_squares(prob.fun, x0=x0, jac=prob.jac, method=method, verbose=2)
        print(res)
        np.testing.assert_allclose(res.x, [3, 2], rtol=1e-14)
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
        verbose=2,
    )
    np.testing.assert_allclose(res.x, [1, 1], rtol=1e-14)
    assert res.success

    prob = ExponentialFittingProblem(a=3, b=2, noise=0.0)
    x0 = np.zeros(2)
    res = trf_no_bounds(
        prob.fun,
        prob.jac,
        x0,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100,
        x_scale=1.0,
        verbose=2,
    )
    print(res)
    np.testing.assert_allclose(res.x, [3, 2], rtol=1e-14)
    assert res.success
