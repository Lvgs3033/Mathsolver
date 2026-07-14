// ---------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
    panels.forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------
async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || "Something went wrong.");
  }
  return data;
}

function renderSteps(container, title, data, resultLabel) {
  const stepsHtml = (data.steps || [])
    .map(s => `<li>${escapeHtml(s)}</li>`)
    .join("");

  const resultText = Array.isArray(data.result) ? data.result.join(", ") :
    (typeof data.result === "object" ? JSON.stringify(data.result) : data.result);

  container.innerHTML = `
    <h2>${title}</h2>
    <ol class="step-list">${stepsHtml}</ol>
    <div class="final-answer">
      <span class="label">${resultLabel}</span>
      ${escapeHtml(String(resultText))}
    </div>
  `;
}

function renderError(container, title, message) {
  container.innerHTML = `
    <h2>${title}</h2>
    <div class="error-box">${escapeHtml(message)}</div>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setLoading(btn, isLoading, label = "Solve") {
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "Computing…" : label;
}

// ---------------------------------------------------------------
// ALGEBRA
// ---------------------------------------------------------------
const algOp = document.getElementById("alg-op");
const algVarRow = document.getElementById("alg-var-row");
const algRunBtn = document.getElementById("alg-run");
const algResult = document.getElementById("alg-result");

algOp.addEventListener("change", () => {
  algVarRow.style.display = algOp.value === "solve" ? "" : "none";
});

algRunBtn.addEventListener("click", async () => {
  setLoading(algRunBtn, true);
  try {
    const op = algOp.value;
    const expr = document.getElementById("alg-input").value;
    let data;
    if (op === "solve") {
      data = await postJSON("/api/algebra/solve", {
        equation: expr,
        variable: document.getElementById("alg-var").value || "x",
      });
      renderSteps(algResult, "Steps", data, "Solution");
    } else {
      data = await postJSON("/api/algebra/op", { op, expression: expr });
      renderSteps(algResult, "Steps", data, "Result");
    }
  } catch (e) {
    renderError(algResult, "Steps", e.message);
  } finally {
    setLoading(algRunBtn, false, "Solve");
  }
});

// ---------------------------------------------------------------
// CALCULUS
// ---------------------------------------------------------------
const calcOp = document.getElementById("calc-op");
const calcOrderRow = document.getElementById("calc-order-row");
const calcBoundsRow = document.getElementById("calc-bounds-row");
const calcPointRow = document.getElementById("calc-point-row");
const calcDirRow = document.getElementById("calc-dir-row");
const calcRunBtn = document.getElementById("calc-run");
const calcResult = document.getElementById("calc-result");

function refreshCalcFields() {
  const op = calcOp.value;
  calcOrderRow.classList.toggle("hidden", op !== "derivative");
  calcBoundsRow.classList.toggle("hidden", op !== "integral");
  calcPointRow.classList.toggle("hidden", op !== "limit");
  calcDirRow.classList.toggle("hidden", op !== "limit");
}
calcOp.addEventListener("change", refreshCalcFields);
refreshCalcFields();

calcRunBtn.addEventListener("click", async () => {
  setLoading(calcRunBtn, true);
  try {
    const op = calcOp.value;
    const expression = document.getElementById("calc-input").value;
    const variable = document.getElementById("calc-var").value || "x";
    let data;

    if (op === "derivative") {
      data = await postJSON("/api/calculus/derivative", {
        expression, variable, order: document.getElementById("calc-order").value,
      });
      renderSteps(calcResult, "Steps", data, "Derivative");
    } else if (op === "integral") {
      data = await postJSON("/api/calculus/integral", {
        expression, variable,
        lower: document.getElementById("calc-lower").value,
        upper: document.getElementById("calc-upper").value,
      });
      renderSteps(calcResult, "Steps", data, "Integral");
    } else {
      data = await postJSON("/api/calculus/limit", {
        expression, variable,
        point: document.getElementById("calc-point").value,
        direction: document.getElementById("calc-dir").value,
      });
      renderSteps(calcResult, "Steps", data, "Limit");
    }
  } catch (e) {
    renderError(calcResult, "Steps", e.message);
  } finally {
    setLoading(calcRunBtn, false, "Solve");
  }
});

// ---------------------------------------------------------------
// MATRICES
// ---------------------------------------------------------------
const matOp = document.getElementById("mat-op");
const matARows = document.getElementById("matA-rows");
const matACols = document.getElementById("matA-cols");
const matBRows = document.getElementById("matB-rows");
const matBCols = document.getElementById("matB-cols");
const matAGrid = document.getElementById("matA-grid");
const matBGrid = document.getElementById("matB-grid");
const matBSection = document.getElementById("matB-section");
const matRunBtn = document.getElementById("mat-run");
const matResult = document.getElementById("mat-result");

const NEEDS_B = new Set(["add", "subtract", "multiply"]);
const NEEDS_SQUARE_A = new Set(["determinant", "inverse", "eigenvalues"]);

function buildGrid(gridEl, rows, cols, fillDiag) {
  gridEl.innerHTML = "";
  gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const inp = document.createElement("input");
      inp.type = "text";
      inp.value = fillDiag ? (r === c ? "1" : "0") : "0";
      inp.dataset.row = r;
      inp.dataset.col = c;
      gridEl.appendChild(inp);
    }
  }
}

function readGrid(gridEl, rows, cols) {
  const values = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const inp = gridEl.querySelector(`input[data-row="${r}"][data-col="${c}"]`);
      row.push(inp.value || "0");
    }
    values.push(row);
  }
  return values;
}

function rebuildA() {
  buildGrid(matAGrid, parseInt(matARows.value) || 1, parseInt(matACols.value) || 1, true);
}
function rebuildB() {
  buildGrid(matBGrid, parseInt(matBRows.value) || 1, parseInt(matBCols.value) || 1, true);
}
[matARows, matACols].forEach(el => el.addEventListener("change", rebuildA));
[matBRows, matBCols].forEach(el => el.addEventListener("change", rebuildB));

function refreshMatFields() {
  const op = matOp.value;
  matBSection.classList.toggle("hidden", !NEEDS_B.has(op));
}
matOp.addEventListener("change", refreshMatFields);

rebuildA();
rebuildB();
refreshMatFields();

matRunBtn.addEventListener("click", async () => {
  setLoading(matRunBtn, true, "Compute");
  try {
    const op = matOp.value;
    const matrixA = readGrid(matAGrid, parseInt(matARows.value), parseInt(matACols.value));
    let payload = { op, matrixA };
    if (NEEDS_B.has(op)) {
      payload.matrixB = readGrid(matBGrid, parseInt(matBRows.value), parseInt(matBCols.value));
    }
    const data = await postJSON("/api/matrix", payload);
    renderSteps(matResult, "Steps", data, "Result");
  } catch (e) {
    renderError(matResult, "Steps", e.message);
  } finally {
    setLoading(matRunBtn, false, "Compute");
  }
});

// ---------------------------------------------------------------
// GRAPH
// ---------------------------------------------------------------
const graphRunBtn = document.getElementById("graph-run");
const graphResult = document.getElementById("graph-result");

graphRunBtn.addEventListener("click", async () => {
  setLoading(graphRunBtn, true, "Plot");
  try {
    const expressions = [
      document.getElementById("graph-f1").value,
      document.getElementById("graph-f2").value,
      document.getElementById("graph-f3").value,
    ];
    const data = await postJSON("/api/plot", {
      expressions,
      xMin: document.getElementById("graph-xmin").value,
      xMax: document.getElementById("graph-xmax").value,
    });
    graphResult.innerHTML = `
      <h2>Graph</h2>
      <div class="graph-img-wrap">
        <img src="data:image/png;base64,${data.image}" alt="Function plot">
      </div>
    `;
  } catch (e) {
    renderError(graphResult, "Graph", e.message);
  } finally {
    setLoading(graphRunBtn, false, "Plot");
  }
});
