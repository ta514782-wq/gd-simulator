"""Core math for the Gradient Descent & Optimization Simulator (MATH 303).

Pure NumPy: no Streamlit here, so every function can be tested on its own.
"""
import numpy as np

TOL = 1e-6          # converged when ||grad f|| < TOL
BLOWUP = 1e8        # declare divergence when ||x|| exceeds this

# Each test function: dim, f, grad, known minimiser (or None), default start,
# plotting domain ((xmin, xmax), (ymin, ymax)), default learning rate.
FUNCS = {
    "1D: f(x) = x^2": dict(
        dim=1,
        f=lambda p: p[0] ** 2,
        g=lambda p: np.array([2 * p[0]]),
        xstar=np.array([0.0]), start=[4.0],
        dom=((-5, 5), None), lr=0.1,
    ),
    "1D: f(x) = x^4 - 3x^2 + x  (two minima)": dict(
        dim=1,
        f=lambda p: p[0] ** 4 - 3 * p[0] ** 2 + p[0],
        g=lambda p: np.array([4 * p[0] ** 3 - 6 * p[0] + 1]),
        xstar=None, start=[2.0],
        dom=((-2.5, 2.5), None), lr=0.02,
    ),
    "2D: bowl f(x,y) = x^2 + 5y^2": dict(
        dim=2,
        f=lambda p: p[0] ** 2 + 5 * p[1] ** 2,
        g=lambda p: np.array([2 * p[0], 10 * p[1]]),
        xstar=np.array([0.0, 0.0]), start=[4.0, 3.0],
        dom=((-5, 5), (-4, 4)), lr=0.1,
    ),
    "2D: Rosenbrock valley": dict(
        dim=2,
        f=lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2,
        g=lambda p: np.array([
            -2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2),
            200 * (p[1] - p[0] ** 2),
        ]),
        xstar=np.array([1.0, 1.0]), start=[-1.2, 1.0],
        dom=((-2, 2), (-1, 3)), lr=0.001,
    ),
    "2D: Himmelblau (4 minima)": dict(
        dim=2,
        f=lambda p: (p[0] ** 2 + p[1] - 11) ** 2 + (p[0] + p[1] ** 2 - 7) ** 2,
        g=lambda p: np.array([
            4 * p[0] * (p[0] ** 2 + p[1] - 11) + 2 * (p[0] + p[1] ** 2 - 7),
            2 * (p[0] ** 2 + p[1] - 11) + 4 * p[1] * (p[0] + p[1] ** 2 - 7),
        ]),
        xstar=None, start=[0.0, 0.0],
        dom=((-5, 5), (-5, 5)), lr=0.01,
    ),
}

METHODS = ["Gradient descent", "Momentum", "Adam"]


def optimize(fn, x0, lr, steps, method="Gradient descent", beta=0.9):
    """Run the iteration x_{k+1} = x_k - lr * (update direction).

    Returns (path, diverged). path is an array of iterates x_0, x_1, ...
    Stops early if the sequence blows up or the gradient is below TOL.
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)            # momentum buffer
    m = np.zeros_like(x)            # Adam first moment
    s = np.zeros_like(x)            # Adam second moment
    path = [x.copy()]
    diverged = False
    with np.errstate(all="ignore"):
        for k in range(1, int(steps) + 1):
            g = fn["g"](x)
            if np.linalg.norm(g) < TOL:
                break
            if method == "Gradient descent":
                x = x - lr * g
            elif method == "Momentum":
                v = beta * v + g
                x = x - lr * v
            else:  # Adam
                m = 0.9 * m + 0.1 * g
                s = 0.999 * s + 0.001 * g * g
                mh = m / (1 - 0.9 ** k)
                sh = s / (1 - 0.999 ** k)
                x = x - lr * mh / (np.sqrt(sh) + 1e-8)
            if (not np.all(np.isfinite(x))) or np.linalg.norm(x) > BLOWUP:
                diverged = True
                break
            path.append(x.copy())
    return np.array(path), diverged


def metrics(fn, path):
    """f(x_k), ||grad f(x_k)||, and ||x_k - x*|| (None if x* unknown)."""
    fvals = np.array([fn["f"](p) for p in path])
    gnorms = np.array([np.linalg.norm(fn["g"](p)) for p in path])
    errs = None
    if fn["xstar"] is not None:
        errs = np.linalg.norm(path - fn["xstar"], axis=1)
    return fvals, gnorms, errs


def hessian(fn, x, h=1e-5):
    """Central-difference Hessian built from the analytic gradient."""
    x = np.array(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        H[:, i] = (fn["g"](x + e) - fn["g"](x - e)) / (2 * h)
    return (H + H.T) / 2


def local_lipschitz(fn, x):
    """Largest |eigenvalue| of the Hessian at x: a local Lipschitz constant
    of the gradient. Plain gradient descent is stable when lr < 2/L."""
    return float(np.max(np.abs(np.linalg.eigvalsh(hessian(fn, x)))))


def theoretical_rate(fn, lr):
    """Linear contraction factor of gradient descent near the minimiser:
    rho = max_i |1 - lr * lambda_i|, lambda_i = eigenvalues of Hessian(x*).
    The iteration converges (locally) iff rho < 1."""
    if fn["xstar"] is None:
        return None
    lam = np.linalg.eigvalsh(hessian(fn, fn["xstar"]))
    return float(np.max(np.abs(1 - lr * lam)))


def observed_rate(errs, window=20):
    """Average ratio e_{k+1}/e_k over the last `window` steps."""
    if errs is None or len(errs) < 3:
        return None
    e = errs[-(window + 1):]
    e = e[e > 1e-300]
    if len(e) < 2:
        return None
    return float(np.exp(np.mean(np.diff(np.log(e)))))


def status(path, diverged, fn):
    g = np.linalg.norm(fn["g"](path[-1]))
    if diverged:
        return "Diverged"
    if g < TOL:
        return "Converged"
    return "Not converged yet"


def iterations_to_tol(fn, path):
    for k, p in enumerate(path):
        if np.linalg.norm(fn["g"](p)) < TOL:
            return k
    return None
