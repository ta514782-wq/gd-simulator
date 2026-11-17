"""AI Gradient Descent & Optimization Simulator
MATH 303: Real Analysis I (Digital Artifact #22)

Run locally:   streamlit run app.py
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import gd_core as core

st.set_page_config(page_title="Gradient Descent Simulator", layout="wide")
st.title("Gradient Descent & Optimization Simulator")
st.caption("MATH 303 Real Analysis I: watching a sequence converge to a minimum")

# ------------------------------------------------------------------ sidebar
name = st.sidebar.selectbox("Function f", list(core.FUNCS))
fn = core.FUNCS[name]
method = st.sidebar.selectbox("Optimizer", core.METHODS)

st.sidebar.markdown("**Starting point x\u2080**")
x0 = [st.sidebar.number_input("x\u2080", value=float(fn["start"][0]),
                              key=f"x0_{name}", step=0.5)]
if fn["dim"] == 2:
    x0.append(st.sidebar.number_input("y\u2080", value=float(fn["start"][1]),
                                      key=f"y0_{name}", step=0.5))

lr = st.sidebar.number_input("Learning rate \u03b7", value=float(fn["lr"]),
                             min_value=1e-5, max_value=5.0, format="%.5f",
                             step=float(fn["lr"]) / 4, key=f"lr_{name}")
steps = st.sidebar.slider("Max iterations", 1, 5000, 200, key=f"n_{name}")

path, diverged = core.optimize(fn, x0, lr, steps, method)
fvals, gnorms, errs = core.metrics(fn, path)
state = core.status(path, diverged, fn)

tab1, tab2, tab3 = st.tabs(["Simulator", "Real Analysis behind it", "Test cases"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", state)
    c2.metric("Iterations used", len(path) - 1)
    c3.metric("f(x_k) at end", f"{fvals[-1]:.6g}")
    c4.metric("||grad f|| at end", f"{gnorms[-1]:.3g}")

    if diverged:
        st.error("The sequence x_k is unbounded: the learning rate is too large "
                 "for this function. See the Real Analysis tab for the exact bound.")
    elif state == "Converged":
        st.success(f"Gradient norm fell below {core.TOL:g}: x_k has converged "
                   f"to {np.round(path[-1], 5)}.")

    left, right = st.columns(2)

    # --- path plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    (xmin, xmax), ydom = fn["dom"]
    if fn["dim"] == 1:
        xs = np.linspace(xmin, xmax, 400)
        ax.plot(xs, [fn["f"]([t]) for t in xs], color="tab:blue")
        px = path[:, 0]
        inside = (px >= xmin) & (px <= xmax)
        ax.plot(px[inside], fvals[inside], "o-", color="crimson", ms=4, lw=1)
        ax.plot(px[0], fvals[0], "s", color="green", label="start")
        ax.plot(px[-1], fvals[-1], "*", color="gold", ms=14, mec="k", label="end")
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    else:
        ymin, ymax = ydom
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
        Z = fn["f"]([X, Y])
        ax.contour(X, Y, Z, levels=np.geomspace(0.1, Z.max(), 25), cmap="viridis")
        ax.plot(path[:, 0], path[:, 1], "o-", color="crimson", ms=3, lw=1)
        ax.plot(*path[0], "s", color="green", label="start")
        ax.plot(*path[-1], "*", color="gold", ms=14, mec="k", label="end")
        if fn["xstar"] is not None:
            ax.plot(*fn["xstar"], "x", color="black", ms=10, label="true minimum")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{method} path"); ax.legend(loc="best")
    left.pyplot(fig)

    # --- convergence plot
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ks = np.arange(len(path))
    ax2.semilogy(ks, np.maximum(gnorms, 1e-16), label="||grad f(x_k)||")
    if errs is not None:
        ax2.semilogy(ks, np.maximum(errs, 1e-16), label="||x_k - x*||")
    ax2.set_xlabel("iteration k"); ax2.set_title("Convergence of the sequence")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    right.pyplot(fig2)

    with st.expander("Show iterates (the sequence x_k)"):
        st.dataframe({"k": ks, **{f"x{i}": path[:, i] for i in range(fn["dim"])},
                      "f(x_k)": fvals, "||grad||": gnorms})

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("1. The iteration is a sequence")
    st.markdown(r"""
Gradient descent builds a sequence by the rule
$x_{k+1} = x_k - \eta\,\nabla f(x_k)$.
"Converges to a minimum" means exactly the Real Analysis statement
$\lim_{k\to\infty} x_k = x^*$, i.e. for every $\varepsilon>0$ there is $N$
with $\|x_k-x^*\|<\varepsilon$ for all $k\ge N$.
""")
    st.subheader("2. Derivatives drive the update")
    st.markdown(r"""
At a minimum of a differentiable function, $\nabla f(x^*)=0$ (Fermat's theorem,
generalised to $\mathbb{R}^n$ through the **partial derivatives**). The
algorithm stops when $\|\nabla f(x_k)\|<10^{-6}$.
""")
    st.subheader("3. Contraction and the learning-rate bound")
    st.markdown(r"""
For the quadratic $f(x)=x^2$ the rule becomes $x_{k+1}=(1-2\eta)x_k$, so
$x_k=(1-2\eta)^k x_0$, a **geometric sequence**. It tends to $0$ iff
$|1-2\eta|<1$, i.e. $0<\eta<1$. In general, near the minimum

$$\rho=\max_i |1-\eta\lambda_i|,\qquad \lambda_i=\text{eigenvalues of the Hessian at }x^*$$

and the iteration converges iff $\rho<1$ (a **contraction mapping** argument).
If $\nabla f$ is Lipschitz with constant $L$, then $\eta<2/L$ is the safe range.
""")

    st.subheader("Live numbers for your current settings")
    x_end = path[-1]
    L = core.local_lipschitz(fn, x_end) if not diverged else core.local_lipschitz(fn, path[0])
    a, b, c = st.columns(3)
    a.metric("Local L (Hessian norm)", f"{L:.4g}")
    b.metric("Safe bound 2/L", f"{2 / L:.4g}" if L > 0 else "inf")
    b_ok = L > 0 and lr < 2 / L
    c.metric("\u03b7 below bound?", "Yes" if b_ok else "No")

    if method == "Gradient descent":
        rho = core.theoretical_rate(fn, lr)
        rho_obs = core.observed_rate(errs)
        if rho is not None:
            d, e = st.columns(2)
            d.metric("Theoretical \u03c1 = max|1 - \u03b7\u03bb\u1d62|", f"{rho:.4f}")
            e.metric("Observed ratio e(k+1)/e(k)",
                     f"{rho_obs:.4f}" if rho_obs is not None else "n/a")
            if rho >= 1:
                st.warning("\u03c1 \u2265 1: the sequence is not a contraction, so it cannot converge.")
            else:
                st.info("\u03c1 < 1: each step shrinks the error by about this factor, "
                        "so the error is dominated by the geometric sequence \u03c1\u1d4f.")
        else:
            st.info("This function has several minima, so there is no single \u03c1. "
                    "Try different starting points and see which minimum the sequence picks.")
    else:
        st.info("The closed-form \u03c1 applies to plain gradient descent. "
                "Switch the optimizer to Gradient descent to compare theory and experiment.")

    st.subheader("4. Limitations")
    st.markdown("""
- The bounds above are exact for quadratics and only **local** for other functions.
- Rosenbrock is ill-conditioned: plain gradient descent needs a tiny \u03b7 and thousands of steps.
- Himmelblau and the 1D quartic have several minima; the limit depends on x\u2080.
- Momentum and Adam are shown for comparison; their convergence proofs are beyond this course.
- This is an educational simulator, not a production optimizer.
""")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Preset test runs (gradient descent)")
    presets = [
        ("2D: bowl f(x,y) = x^2 + 5y^2", 0.1, 500, "Good \u03b7: converges fast"),
        ("2D: bowl f(x,y) = x^2 + 5y^2", 0.01, 500, "Too small \u03b7: converges slowly"),
        ("2D: bowl f(x,y) = x^2 + 5y^2", 0.25, 500, "Too large \u03b7: diverges (\u03b7 > 2/L = 0.2)"),
        ("2D: Rosenbrock valley", 0.001, 5000, "Hard valley: slow, ill-conditioned"),
    ]
    rows = []
    for fname, eta, n, note in presets:
        f_ = core.FUNCS[fname]
        p, dv = core.optimize(f_, f_["start"], eta, n)
        fv, gn, _ = core.metrics(f_, p)
        it = core.iterations_to_tol(f_, p)
        rho = core.theoretical_rate(f_, eta)
        rows.append({
            "Function": fname, "\u03b7": eta, "Result": core.status(p, dv, f_),
            "Iterations to tol": it if it is not None else "-",
            "Final f": f"{fv[-1]:.3g}", "Theoretical \u03c1": f"{rho:.4f}",
            "Meaning": note,
        })
    st.table(rows)
    st.caption("These runs cover three behaviours (converge, slow, diverge) plus a hard "
               "real-world-style case, satisfying the 'three input sets' requirement.")
