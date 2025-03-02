import numpy as np

TERMINATION_MESSAGES = {
    -2: "Stopped because `callback` function raised `StopIteration` or returned `True`",
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


def print_header_nonlinear():
    print(
        "{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}".format(
            "Iteration",
            "Total nfev",
            "Cost",
            "Cost reduction",
            "Step norm",
            "Optimality",
        )
    )


def print_iteration_nonlinear(
    iteration, nfev, cost, cost_reduction, step_norm, optimality
):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(
        f"{iteration:^15}{nfev:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}"
    )


def compute_jac_scale(J, scale_inv_old=None):
    """Compute variables scale based on the Jacobian matrix."""
    scale_inv = np.sum(J**2, axis=0) ** 0.5

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


def update_tr_radius(
    Delta, actual_reduction, predicted_reduction, step_norm, bound_hit
):
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
