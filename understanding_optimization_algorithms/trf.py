"""Trust Region Reflective algorithm for least-squares optimization.

The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v:

           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise

where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as

    D^2 g(x) = 0.

Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.

Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies

    (D^2 H + diag(g) Jv) p = -D^2 g

where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying

    B_h p_h = -g_h,

where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables:

    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta

In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is

    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta

Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).

The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).

The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
three: a constrained trust-region step, a reflected step and a constrained
Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
space).

Another feature is that a trust-region radius control strategy is modified to
account for appearance of the diagonal C matrix (called diag_h in the code).

Note that all described peculiarities are completely gone as we consider
problems without bounds (the algorithm becomes a standard trust-region type
algorithm very similar to ones implemented in MINPACK).

The implementation supports two methods of solving the trust-region problem.
The first, called 'exact', applies SVD on Jacobian and then solves the problem
very accurately using the algorithm described in [JJMore]_. It is not
applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
approach (sometimes called "indefinite dogleg"), where the problem is solved
in a subspace spanned by the gradient and the approximate Gauss-Newton step
found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
reformulated as a 4th order algebraic equation and solved very accurately by
``numpy.roots``. The subspace approach allows to solve very large problems
(up to couple of millions of residuals on a regular PC), provided the Jacobian
matrix is sufficiently sparse.

References
----------
.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
      and Conjugate Gradient Method for Large-Scale Bound-Constrained
      Minimization Problems," SIAM Journal on Scientific Computing,
      Vol. 21, Number 1, pp 1-23, 1999.
.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
"""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.optimize import OptimizeResult


EPS = np.finfo(float).eps

TERMINATION_MESSAGES = {
    -2: "Stopped because `callback` function raised `StopIteration` or returned `True`",
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}

def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                           rtol=0.01, max_iter=10):
    """Solve a trust-region problem arising in least-squares minimization.

    This function implements a method described by J. J. More [1]_ and used
    in MINPACK, but it relies on a single SVD of Jacobian instead of series
    of Cholesky decompositions. Before running this function, compute:
    ``U, s, VT = svd(J, full_matrices=False)``.

    Parameters
    ----------
    n : int
        Number of variables.
    m : int
        Number of residuals.
    uf : ndarray
        Computed as U.T.dot(f).
    s : ndarray
        Singular values of J.
    V : ndarray
        Transpose of VT.
    Delta : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.
    n_iter : int
        Number of iterations made by root-finding procedure. Zero means
        that Gauss-Newton step was selected as the solution.

    References
    ----------
    .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
           and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
           in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    """
    def phi_and_derivative(alpha, suf, s, Delta):
        """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `Delta`". Refer to [1]_.
        """
        denom = s**2 + alpha
        p_norm = norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -np.sum(suf ** 2 / denom**3) / p_norm
        return phi, phi_prime

    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    if m >= n:
        threshold = EPS * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return p, 0.0, 0

    alpha_upper = norm(suf) / Delta

    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
    else:
        alpha = initial_alpha

    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)

        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta

        if np.abs(phi) < rtol * Delta:
            break

    p = -V.dot(suf / (s**2 + alpha))

    # Make the norm of p equal to Delta, p is changed only slightly during
    # this. It is done to prevent p lie outside the trust region (which can
    # cause problems later).
    p *= Delta / norm(p)

    return p, alpha, it + 1


def evaluate_quadratic(J, g, s, diag=None):
    """Compute values of a quadratic function arising in least squares.

    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : ndarray, sparse array or LinearOperator, shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (k, n) or (n,)
        Array containing steps as rows.
    diag : ndarray, shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.

    Returns
    -------
    values : ndarray with shape (k,) or float
        Values of the function. If `s` was 2-D, then ndarray is
        returned, otherwise, float is returned.
    """
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.dot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    l = np.dot(s, g)

    return 0.5 * q + l


def compute_grad(J, f):
    """Compute gradient of the least-squares cost function."""
    # if isinstance(J, LinearOperator):
    #     return J.rmatvec(f)
    # else:
    #     return J.T.dot(f)
    return J.T.dot(f)


def compute_jac_scale(J, scale_inv_old=None):
    """Compute variables scale based on the Jacobian matrix."""
    # if issparse(J):
    #     scale_inv = np.asarray(J.power(2).sum(axis=0)).ravel()**0.5
    # else:
    #     scale_inv = np.sum(J**2, axis=0)**0.5
    scale_inv = np.sum(J**2, axis=0)**0.5

    if scale_inv_old is None:
        scale_inv[scale_inv == 0] = 1
    else:
        scale_inv = np.maximum(scale_inv, scale_inv_old)

    return 1 / scale_inv, scale_inv


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """Check termination condition for nonlinear least squares."""
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None
    

def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm, bound_hit):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0

    return Delta, ratio


def print_header_nonlinear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality"))


def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(f"{iteration:^15}{nfev:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}")


def trf_no_bounds(fun, jac, x0, ftol, xtol, gtol, max_nfev, x_scale, verbose):
    x = x0.copy()

    f0 = fun(x0)
    J0 = jac(x0)
    initial_cost = 0.5 * np.dot(f0, f0)

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == "jac"
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter

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

        d = scale
        g_h = d * g

        J_h = J * d
        U, s, V = svd(J_h, full_matrices=False)
        V = V.T
        uf = U.T.dot(f)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            step_h, alpha, n_iter = solve_lsq_trust_region(
                n, m, uf, s, V, Delta, initial_alpha=alpha
            )

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                step_h_norm > 0.95 * Delta,
            )

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
            )
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x)
            njev += 1

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = np.zeros_like(x)
    result = OptimizeResult(
        x=x,
        cost=cost,
        fun=f_true,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=termination_status,
    )

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    if verbose >= 1:
        print(result.message)
        print(
            f"Function evaluations {result.nfev}, initial cost {initial_cost:.4e}, "
            f"final cost {result.cost:.4e}, "
            f"first-order optimality {result.optimality:.2e}."
        )

    return result
