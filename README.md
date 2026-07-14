# Solvit — Math Problem Solver

**Preview:** https://mathsolver-2.onrender.com/
**Docker Image:** docker pull dhvanik23/mathsolver 

A full-stack math problem solver: **Flask** backend, **HTML/CSS/JS** frontend,
**SymPy** for exact symbolic algebra/calculus/matrices, **Matplotlib** for graph plotting.

Every result is computed exactly with SymPy (a real computer-algebra system) —
nothing is hard-coded, guessed, or randomised. Step explanations are generated
programmatically from the actual structure of your expression, so they always
match the real computation.

## Features

- **Algebra**: solve equations (linear, quadratic, polynomial, etc.), simplify, factor, expand
- **Calculus**: derivatives (any order, with product/chain/power rule breakdown),
  indefinite & definite integrals, limits (with indeterminate-form detection)
- **Matrices**: add, subtract, multiply, determinant, inverse, transpose, eigenvalues, rank
- **Graph plotting**: plot up to 3 functions on one chart with a custom x-range

## Tech stack

| Layer      | Technology              |
|------------|--------------------------|
| Backend    | Flask (Python)           |
| Math engine| SymPy (exact symbolic math) |
| Plotting   | Matplotlib + NumPy       |
| Frontend   | HTML5, CSS3, vanilla JavaScript (fetch API) |

## Setup

```bash
cd mathsolver
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Project structure

```
mathsolver/
├── app.py                 # Flask app + SymPy/Matplotlib logic (all API routes)
├── requirements.txt
├── templates/
│   └── index.html         # Single-page UI (4 tabs)
└── static/
    ├── style.css          # Design system (graph-paper / chalkboard theme)
    └── script.js          # Tab logic + fetch calls to the API
```

## API reference

All endpoints accept/return JSON.

| Endpoint                     | Body                                                        |
|-------------------------------|--------------------------------------------------------------|
| `POST /api/algebra/solve`     | `{ equation, variable }`                                     |
| `POST /api/algebra/op`        | `{ op: simplify\|factor\|expand, expression }`                |
| `POST /api/calculus/derivative` | `{ expression, variable, order }`                           |
| `POST /api/calculus/integral` | `{ expression, variable, lower?, upper? }`                    |
| `POST /api/calculus/limit`    | `{ expression, variable, point, direction }`                  |
| `POST /api/matrix`            | `{ op, matrixA, matrixB? }`                                   |
| `POST /api/plot`              | `{ expressions: [...], xMin, xMax }`                          |

Every response includes `"ok": true/false`. On success you get `steps` (array of
strings) and `result`; on failure you get a descriptive `error` string — inputs
are validated with SymPy's real parser, so malformed expressions are rejected
with an explanit error rather than silently producing a wrong answer.

## Notes on accuracy

- All parsing goes through `sympy.parsing.sympy_parser` with a restricted symbol
  table (no `eval`, no arbitrary code execution).
- Derivatives/integrals/limits/matrix results are always cross-checked against
  SymPy's own `diff`, `integrate`, `limit`, and `Matrix` methods — the step text
  describes the same computation SymPy performs, it doesn't just narrate a guess.
- Graphs are rendered by evaluating the exact SymPy expression (via `lambdify`)
  over a NumPy array, so the curve matches the symbolic function precisely.
