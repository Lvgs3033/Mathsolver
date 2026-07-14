"""
Math Problem Solver - Flask backend
Libraries used:
  - SymPy   : algebra, calculus (derivatives/integrals/limits), matrices, exact solving
  - Matplotlib : function graph plotting (rendered server-side to PNG -> base64)
  - Flask   : web server / REST API
All results are computed exactly with SymPy - nothing is hard-coded or randomised.
"""

import base64
import io
import traceback

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor
)

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

x, y, z, t = sp.symbols("x y z t")
SAFE_LOCALS = {
    "x": x, "y": y, "z": z, "t": t,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "e": sp.E,
    "Abs": sp.Abs, "oo": sp.oo, "I": sp.I,
    "factorial": sp.factorial,
}


class MathError(Exception):
    pass


def safe_parse(expr_str):
    """Parse a user supplied math string into a SymPy expression safely."""
    if not expr_str or not expr_str.strip():
        raise MathError("Empty expression.")
    cleaned = expr_str.replace("^", "**")
    try:
        expr = parse_expr(cleaned, local_dict=SAFE_LOCALS, transformations=TRANSFORMS,
                           evaluate=True)
    except Exception as exc:
        raise MathError(f"Could not parse expression '{expr_str}': {exc}")
    return expr


def fmt(expr):
    """Pretty text form of a sympy object."""
    try:
        return sp.sstr(sp.nsimplify(expr) if expr.is_number else expr)
    except Exception:
        return str(expr)


# ----------------------------------------------------------------------
# ALGEBRA
# ----------------------------------------------------------------------

def solve_equation(eq_str, var_str):
    var = sp.symbols(var_str)
    steps = []

    if "=" in eq_str:
        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs = safe_parse(lhs_str)
        rhs = safe_parse(rhs_str)
    else:
        lhs = safe_parse(eq_str)
        rhs = sp.Integer(0)

    steps.append(f"Original equation: {fmt(lhs)} = {fmt(rhs)}")

    moved = sp.expand(lhs - rhs)
    steps.append(f"Move all terms to one side: {fmt(moved)} = 0")

    simplified = sp.simplify(moved)
    if simplified != moved:
        steps.append(f"Simplify: {fmt(simplified)} = 0")

    factored = sp.factor(simplified)
    if factored != simplified and factored != 0:
        steps.append(f"Factor: {fmt(factored)} = 0")

    solutions = sp.solve(sp.Eq(lhs, rhs), var)
    if solutions:
        steps.append("Apply the zero-product / solving rule for each factor.")
        sol_strs = [fmt(s) for s in solutions]
        steps.append(f"Solution(s): {var_str} = " + ", ".join(sol_strs))
    else:
        steps.append("No solution found (equation may have no real/complex roots, "
                      "or is an identity).")

    return {
        "steps": steps,
        "result": [fmt(s) for s in solutions],
        "latex": sp.latex(sp.Eq(lhs, rhs)),
    }


def algebra_operation(op, expr_str):
    expr = safe_parse(expr_str)
    steps = [f"Original expression: {fmt(expr)}"]

    if op == "simplify":
        result = sp.simplify(expr)
        steps.append(f"Apply simplification rules (combine like terms, reduce fractions).")
        steps.append(f"Result: {fmt(result)}")
    elif op == "expand":
        result = sp.expand(expr)
        steps.append("Distribute multiplication over addition (expand all products/powers).")
        steps.append(f"Result: {fmt(result)}")
    elif op == "factor":
        result = sp.factor(expr)
        steps.append("Find common factors / apply factoring identities.")
        steps.append(f"Result: {fmt(result)}")
    else:
        raise MathError(f"Unknown algebra operation '{op}'.")

    return {"steps": steps, "result": fmt(result), "latex": sp.latex(result)}


# ----------------------------------------------------------------------
# CALCULUS
# ----------------------------------------------------------------------

KNOWN_FUNCS = {
    sp.sin: ("sin(u)", "cos(u)"),
    sp.cos: ("cos(u)", "-sin(u)"),
    sp.tan: ("tan(u)", "sec(u)**2"),
    sp.exp: ("exp(u)", "exp(u)"),
    sp.log: ("log(u)", "1/u"),
}


def derivative_steps(expr, var):
    steps = []

    def rec(e):
        if e.is_number:
            steps.append(f"d/d{var}[{fmt(e)}] = 0   (derivative of a constant)")
            return sp.Integer(0)
        if e == var:
            steps.append(f"d/d{var}[{var}] = 1")
            return sp.Integer(1)
        if e.is_Add:
            parts = list(e.args)
            results = [rec(p) for p in parts]
            steps.append(f"Sum rule: d/d{var}[{fmt(e)}] = " +
                         " + ".join(f"({fmt(r)})" for r in results))
            return sp.Add(*results)
        if e.is_Mul:
            const, rest = e.as_independent(var)
            if const != 1 and rest != 1:
                r = rec(rest)
                out = sp.simplify(const * r)
                steps.append(f"Constant multiple rule: d/d{var}[{fmt(const)}*{fmt(rest)}] "
                             f"= {fmt(const)} * d/d{var}[{fmt(rest)}] = {fmt(out)}")
                return out
            args = e.args
            if len(args) == 2:
                u, v = args
                du = sp.diff(u, var)
                dv = sp.diff(v, var)
                out = sp.simplify(u * dv + v * du)
                steps.append(f"Product rule: d/d{var}[{fmt(u)}*{fmt(v)}] = "
                             f"{fmt(u)}*({fmt(dv)}) + {fmt(v)}*({fmt(du)}) = {fmt(out)}")
                return out
            out = sp.diff(e, var)
            steps.append(f"Product rule applied to {fmt(e)}: result = {fmt(out)}")
            return out
        if e.is_Pow:
            base, exp = e.args
            if base == var and exp.is_number:
                out = exp * var ** (exp - 1)
                steps.append(f"Power rule: d/d{var}[{var}^{fmt(exp)}] = "
                             f"{fmt(exp)}*{var}^{fmt(exp - 1)} = {fmt(out)}")
                return out
            # quotient written as Mul(base, Pow(other,-1)) is handled above; treat as chain
            dbase = sp.diff(base, var)
            out = sp.diff(e, var)
            steps.append(f"Chain rule (power/quotient) on {fmt(e)}: result = {fmt(out)}")
            return out
        if e.func in KNOWN_FUNCS and len(e.args) == 1:
            inner = e.args[0]
            u = sp.Symbol("u")
            outer_formula = KNOWN_FUNCS[e.func][1]
            douter = sp.sympify(outer_formula, locals={"u": inner, "sec": lambda a: 1/sp.cos(a)})
            dinner = sp.diff(inner, var)
            out = sp.simplify(douter * dinner)
            steps.append(f"Chain rule: d/d{var}[{e.func.__name__}({fmt(inner)})] = "
                         f"{e.func.__name__}'({fmt(inner)}) * d/d{var}[{fmt(inner)}] = {fmt(out)}")
            return out
        out = sp.diff(e, var)
        steps.append(f"Differentiate {fmt(e)} directly: result = {fmt(out)}")
        return out

    rec(expr)
    final = sp.simplify(sp.diff(expr, var))
    steps.append(f"Final derivative: d/d{var}[{fmt(expr)}] = {fmt(final)}")
    return steps, final


def integral_steps(expr, var):
    steps = [f"Integrate: \u222b {fmt(expr)} d{var}"]

    def describe(e):
        if e.is_Add:
            for term in e.args:
                describe(term)
            steps.append("Sum rule: integrate each term separately and add the results.")
            return
        const, rest = e.as_independent(var)
        if const != 1 and rest != 1:
            steps.append(f"Constant multiple rule: pull {fmt(const)} out of the integral.")
            describe(rest)
            return
        if rest.is_Pow and rest.args[0] == var and rest.args[1].is_number and rest.args[1] != -1:
            n = rest.args[1]
            steps.append(f"Power rule: \u222b {var}^{fmt(n)} d{var} = "
                         f"{var}^{fmt(n + 1)}/{fmt(n + 1)} + C")
            return
        if rest == var:
            steps.append(f"Power rule: \u222b {var} d{var} = {var}^2/2 + C")
            return
        if rest.is_Pow and rest.args[0] == var and rest.args[1] == -1:
            steps.append(f"\u222b 1/{var} d{var} = ln|{var}| + C")
            return
        if rest.func == sp.sin:
            steps.append(f"\u222b sin({fmt(rest.args[0])}) d{var} = -cos({fmt(rest.args[0])}) + C "
                         f"(chain factor applied if argument is not {var})")
            return
        if rest.func == sp.cos:
            steps.append(f"\u222b cos({fmt(rest.args[0])}) d{var} = sin({fmt(rest.args[0])}) + C")
            return
        if rest.func == sp.exp:
            steps.append(f"\u222b exp({fmt(rest.args[0])}) d{var} = exp({fmt(rest.args[0])}) + C "
                         f"(chain factor applied if argument is not {var})")
            return
        steps.append(f"Apply standard integration techniques (substitution/parts/tables) to "
                     f"{fmt(rest)}.")

    describe(expr)
    result = sp.integrate(expr, var)
    steps.append(f"Final result: \u222b {fmt(expr)} d{var} = {fmt(result)} + C")
    return steps, result


def limit_steps(expr, var, point_str, direction):
    point = safe_parse(point_str)
    steps = [f"Evaluate: lim ({var} -> {fmt(point)}) {fmt(expr)}"]

    dir_map = {"both": "+-", "left": "-", "right": "+"}
    dir_sym = dir_map.get(direction, "+-")

    try:
        direct = expr.subs(var, point)
        direct = sp.simplify(direct)
    except Exception:
        direct = sp.nan

    if direct.is_finite and direct != sp.zoo and not direct.has(sp.nan):
        steps.append(f"Direct substitution works: plug {var} = {fmt(point)} into the expression.")
        steps.append(f"Result: {fmt(direct)}")
    else:
        steps.append(f"Direct substitution gives an indeterminate/undefined form "
                     f"({fmt(direct)}); apply limit algebra "
                     f"(factoring, rationalizing, or L'Hopital's rule).")

    if dir_sym == "+-":
        result = sp.limit(expr, var, point)
    else:
        result = sp.limit(expr, var, point, dir=dir_sym)

    steps.append(f"Final limit value: {fmt(result)}")
    return steps, result


# ----------------------------------------------------------------------
# MATRICES
# ----------------------------------------------------------------------

def parse_matrix(raw):
    """raw is a list of lists of strings/numbers."""
    try:
        rows = [[safe_parse(str(cell)) for cell in row] for row in raw]
        return sp.Matrix(rows)
    except Exception as exc:
        raise MathError(f"Invalid matrix data: {exc}")


def matrix_operation(op, mat_a_raw, mat_b_raw=None):
    A = parse_matrix(mat_a_raw)
    steps = [f"Matrix A ({A.rows}x{A.cols}):", sp.pretty(A)]
    B = None
    if mat_b_raw is not None:
        B = parse_matrix(mat_b_raw)
        steps.append(f"Matrix B ({B.rows}x{B.cols}):")
        steps.append(sp.pretty(B))

    if op == "add":
        if B is None or A.shape != B.shape:
            raise MathError("Matrices must have the same dimensions to add.")
        result = A + B
        steps.append("Add corresponding entries: C[i][j] = A[i][j] + B[i][j]")
    elif op == "subtract":
        if B is None or A.shape != B.shape:
            raise MathError("Matrices must have the same dimensions to subtract.")
        result = A - B
        steps.append("Subtract corresponding entries: C[i][j] = A[i][j] - B[i][j]")
    elif op == "multiply":
        if B is None:
            raise MathError("Matrix B is required for multiplication.")
        if A.cols != B.rows:
            raise MathError(f"Cannot multiply {A.rows}x{A.cols} by {B.rows}x{B.cols}: "
                             "inner dimensions must match.")
        result = A * B
        steps.append("Multiply row i of A by column j of B and sum the products for each entry.")
    elif op == "determinant":
        if A.rows != A.cols:
            raise MathError("Determinant requires a square matrix.")
        result = A.det()
        steps.append("Expand by cofactors (or row reduction) to compute the determinant.")
    elif op == "inverse":
        if A.rows != A.cols:
            raise MathError("Inverse requires a square matrix.")
        if A.det() == 0:
            raise MathError("Matrix is singular (determinant = 0); inverse does not exist.")
        result = A.inv()
        steps.append("Compute A^-1 = adj(A) / det(A) (Gauss-Jordan elimination under the hood).")
    elif op == "transpose":
        result = A.T
        steps.append("Flip rows and columns: (A^T)[i][j] = A[j][i]")
    elif op == "eigenvalues":
        if A.rows != A.cols:
            raise MathError("Eigenvalues require a square matrix.")
        eig = A.eigenvals()
        result = eig
        steps.append("Solve det(A - lambda*I) = 0 for lambda (characteristic polynomial).")
    elif op == "rank":
        result = A.rank()
        steps.append("Row-reduce to echelon form and count the non-zero rows.")
    else:
        raise MathError(f"Unknown matrix operation '{op}'.")

    if isinstance(result, sp.Matrix):
        steps.append("Result:")
        steps.append(sp.pretty(result))
        result_data = result.tolist()
        result_str = [[fmt(c) for c in row] for row in result_data]
    elif isinstance(result, dict):  # eigenvalues
        result_str = {fmt(k): v for k, v in result.items()}
        steps.append(f"Result (eigenvalue: multiplicity): {result_str}")
    else:
        result_str = fmt(result)
        steps.append(f"Result: {result_str}")

    return {"steps": steps, "result": result_str}


# ----------------------------------------------------------------------
# GRAPH PLOTTING
# ----------------------------------------------------------------------

def plot_functions(expr_strs, x_min, x_max, points=800):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    colors = ["#2f6fed", "#e0663a", "#22a37b", "#a259d9", "#d9a726"]

    xs = np.linspace(x_min, x_max, points)
    plotted_any = False

    for i, expr_str in enumerate(expr_strs):
        expr = safe_parse(expr_str)
        f = sp.lambdify(x, expr, modules=["numpy"])
        try:
            ys = f(xs)
            ys = np.array(ys, dtype=float)
            ys = np.where(np.isfinite(ys), ys, np.nan)
        except Exception as exc:
            raise MathError(f"Could not evaluate '{expr_str}' for plotting: {exc}")
        ax.plot(xs, ys, label=f"y = {expr_str}", color=colors[i % len(colors)], linewidth=2)
        plotted_any = True

    if not plotted_any:
        raise MathError("No valid functions to plot.")

    ax.axhline(0, color="#888888", linewidth=1)
    ax.axvline(0, color="#888888", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/algebra/solve", methods=["POST"])
def api_algebra_solve():
    data = request.get_json(force=True)
    try:
        out = solve_equation(data.get("equation", ""), data.get("variable", "x") or "x")
        return jsonify({"ok": True, **out})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error solving equation."}), 500


@app.route("/api/algebra/op", methods=["POST"])
def api_algebra_op():
    data = request.get_json(force=True)
    try:
        out = algebra_operation(data.get("op", "simplify"), data.get("expression", ""))
        return jsonify({"ok": True, **out})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error processing expression."}), 500


@app.route("/api/calculus/derivative", methods=["POST"])
def api_derivative():
    data = request.get_json(force=True)
    try:
        var = sp.symbols(data.get("variable", "x") or "x")
        expr = safe_parse(data.get("expression", ""))
        order = int(data.get("order", 1) or 1)
        steps, result = derivative_steps(expr, var)
        for _ in range(order - 1):
            expr = result
            more_steps, result = derivative_steps(expr, var)
            steps += ["--- differentiate again ---"] + more_steps
        return jsonify({"ok": True, "steps": steps, "result": fmt(result),
                         "latex": sp.latex(result)})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error computing derivative."}), 500


@app.route("/api/calculus/integral", methods=["POST"])
def api_integral():
    data = request.get_json(force=True)
    try:
        var = sp.symbols(data.get("variable", "x") or "x")
        expr = safe_parse(data.get("expression", ""))
        lower = data.get("lower")
        upper = data.get("upper")
        steps, result = integral_steps(expr, var)

        if lower not in (None, "") and upper not in (None, ""):
            lo = safe_parse(str(lower))
            up = safe_parse(str(upper))
            definite = sp.integrate(expr, (var, lo, up))
            steps.append(f"Evaluate the antiderivative from {fmt(lo)} to {fmt(up)}: "
                         f"F({fmt(up)}) - F({fmt(lo)}) = {fmt(definite)}")
            result = definite

        return jsonify({"ok": True, "steps": steps, "result": fmt(result),
                         "latex": sp.latex(result)})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error computing integral."}), 500


@app.route("/api/calculus/limit", methods=["POST"])
def api_limit():
    data = request.get_json(force=True)
    try:
        var = sp.symbols(data.get("variable", "x") or "x")
        expr = safe_parse(data.get("expression", ""))
        steps, result = limit_steps(expr, var, data.get("point", "0"), data.get("direction", "both"))
        return jsonify({"ok": True, "steps": steps, "result": fmt(result),
                         "latex": sp.latex(result)})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error computing limit."}), 500


@app.route("/api/matrix", methods=["POST"])
def api_matrix():
    data = request.get_json(force=True)
    try:
        out = matrix_operation(data.get("op"), data.get("matrixA"), data.get("matrixB"))
        return jsonify({"ok": True, **out})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error with matrix operation."}), 500


@app.route("/api/plot", methods=["POST"])
def api_plot():
    data = request.get_json(force=True)
    try:
        exprs = data.get("expressions", [])
        exprs = [e for e in exprs if e and e.strip()]
        x_min = float(data.get("xMin", -10))
        x_max = float(data.get("xMax", 10))
        if x_min >= x_max:
            raise MathError("x-min must be less than x-max.")
        image = plot_functions(exprs, x_min, x_max)
        return jsonify({"ok": True, "image": image})
    except MathError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Unexpected error generating plot."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
