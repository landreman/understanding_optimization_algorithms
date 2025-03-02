"""
This file contains the dogbox algorithm from scipy.optimize,
but with some of the bells and whistles removed to make it easier to understand:
* No box constraints.
* No sparse linear algebra.
* No loss functions.
* No callback functions.
"""

"""
Dogleg algorithm with rectangular trust regions for least-squares minimization.

The description of the algorithm can be found in [Voglis]_. The algorithm does
trust-region iterations, but the shape of trust regions is rectangular as
opposed to conventional elliptical. The intersection of a trust region and
an initial feasible region is again some rectangle. Thus, on each iteration a
bound-constrained quadratic optimization problem is solved.

A quadratic problem is solved by well-known dogleg approach, where the
function is minimized along piecewise-linear "dogleg" path [NumOpt]_,
Chapter 4. If Jacobian is not rank-deficient then the function is decreasing
along this path, and optimization amounts to simply following along this
path as long as a point stays within the bounds. A constrained Cauchy step
(along the anti-gradient) is considered for safety in rank deficient cases,
in this situations the convergence might be slow.

If during iterations some variable hit the initial bound and the component
of anti-gradient points outside the feasible region, then a next dogleg step
won't make any progress. At this state such variables satisfy first-order
optimality conditions and they are excluded before computing a next dogleg
step.

Gauss-Newton step can be computed exactly by `numpy.linalg.lstsq` (for dense
Jacobian matrices) or by iterative procedure `scipy.sparse.linalg.lsmr` (for
dense and sparse matrices, or Jacobian being LinearOperator). The second
option allows to solve very large problems (up to couple of millions of
residuals on a regular PC), provided the Jacobian matrix is sufficiently
sparse. But note that dogbox is not very good for solving problems with
large number of constraints, because of variables exclusion-inclusion on each
iteration (a required number of function evaluations might be high or accuracy
of a solution will be poor), thus its large-scale usage is probably limited
to unconstrained problems.

References
----------
.. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
            Approach for Unconstrained and Bound Constrained Nonlinear
            Optimization", WSEAS International Conference on Applied
            Mathematics, Corfu, Greece, 2004.
.. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition".
"""
import numpy as np
from numpy.linalg import lstsq, norm

from scipy.optimize import OptimizeResult


from .common import (
    TERMINATION_MESSAGES,
    print_header_nonlinear,
    print_iteration_nonlinear,
    check_termination,
    compute_jac_scale,
    update_tr_radius,
    evaluate_quadratic,
    finish_up,
)


def step_size_to_bound(x, s, lb, ub):
    """Compute a min_step size required to reach a bound.

    The function computes a positive scalar t, such that x + s * t is on
    the bound.

    Returns
    -------
    step : float
        Computed step. Non-negative value.
    hits : ndarray of int with shape of x
        Each element indicates whether a corresponding variable reaches the
        bound:

             *  0 - the bound was not hit.
             * -1 - the lower bound was hit.
             *  1 - the upper bound was hit.
    """
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over="ignore"):
        steps[non_zero] = np.maximum(
            (lb - x)[non_zero] / s_non_zero, (ub - x)[non_zero] / s_non_zero
        )
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def build_quadratic_1d(J, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows::

        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)

    Parameters
    ----------
    J : ndarray, sparse array or LinearOperator shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (n,)
        Direction vector of a line.
    diag : None or ndarray with shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.
    s0 : None or ndarray with shape (n,), optional
        Initial point. If None, assumed to be 0.

    Returns
    -------
    a : float
        Coefficient for t**2.
    b : float
        Coefficient for t.
    c : float
        Free term. Returned only if `s0` is provided.
    """
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5

    b = np.dot(g, s)

    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize a 1-D quadratic function subject to bounds.

    The free term `c` is 0 by default. Bounds must be finite.

    Returns
    -------
    t : float
        Minimum point.
    y : float
        Minimum value.
    """
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return t[min_index], y[min_index]


def dogleg_step(x, newton_step, g, a, b, tr_bounds):
    """Find dogleg step in a rectangular region.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    tr_hit : bool
        Whether the step hit the boundary of the trust-region.
    """
    lb_total = -tr_bounds
    ub_total = tr_bounds

    if np.all((newton_step >= lb_total) & (newton_step <= ub_total)):
        return newton_step, False

    to_bounds, _ = step_size_to_bound(np.zeros_like(x), -g, lb_total, ub_total)

    # The classical dogleg algorithm would check if Cauchy step fits into
    # the bounds, and just return it constrained version if not. But in a
    # rectangular trust region it makes sense to try to improve constrained
    # Cauchy step too. Thus, we don't distinguish these two cases.

    cauchy_step = -minimize_quadratic_1d(a, b, 0, to_bounds)[0] * g

    step_diff = newton_step - cauchy_step
    step_size, hits = step_size_to_bound(cauchy_step, step_diff, lb_total, ub_total)
    tr_hit = np.any((hits < 0) | (hits > 0))

    return cauchy_step + step_size * step_diff, tr_hit


def dogbox(fun, jac, x0, ftol, xtol, gtol, max_nfev, x_scale, verbose):
    """This is the main optimization algorithm."""
    f = fun(x0)
    J = jac(x0)
    nfev = 1
    njev = 1
    initial_cost = 0.5 * np.dot(f, f)
    cost = initial_cost.copy()
    if isinstance(x_scale, float):
        x_scale = np.full_like(x0, x_scale)

    # Compute gradient of the least-squares cost function:
    g = J.T.dot(f)

    jac_scale = isinstance(x_scale, str) and x_scale == "jac"
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv, ord=np.inf)
    if Delta == 0:
        Delta = 1.0

    x = x0
    step = np.empty_like(x0)

    if max_nfev is None:
        max_nfev = x0.size * 100

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(
                iteration, nfev, cost, actual_reduction, step_norm, g_norm
            )

        if termination_status is not None or nfev == max_nfev:
            break

        # Compute (Gauss-)Newton and build quadratic model for Cauchy step.
        newton_step = lstsq(J, -f, rcond=-1)[0]

        # Coefficients for the quadratic model along the anti-gradient.
        a, b = build_quadratic_1d(J, g, -g)

        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            tr_bounds = Delta * scale

            step, tr_hit = dogleg_step(
                x, newton_step, g, a, b, tr_bounds
            )

            predicted_reduction = -evaluate_quadratic(J, g, step)

            x_new = x + step

            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step * scale_inv, ord=np.inf)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction, step_h_norm, tr_hit
            )

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
            )

            if termination_status is not None:
                break

        if actual_reduction > 0:

            x = x_new
            f = f_new

            cost = cost_new

            J = jac(x)
            njev += 1

            # Compute gradient of the least-squares cost function:
            g = J.T.dot(f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    # Done with the main iteration. The rest of this function is just tidying up
    # the results.

    if termination_status is None:
        termination_status = 0

    active_mask = np.zeros_like(x)
    result = OptimizeResult(
        x=x,
        cost=cost,
        fun=f,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=termination_status,
    )

    return finish_up(result, initial_cost, verbose)
