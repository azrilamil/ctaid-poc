/* CTAID POC front end. Talks only to this app's /api routes. */
(function () {
  const form = document.getElementById("predict-form");
  const result = document.getElementById("result");
  const metricsEl = document.getElementById("metrics");
  const sampleSelect = document.getElementById("sample-select");
  let schema = null;

  function fieldInput(spec) {
    const wrap = document.createElement("label");
    wrap.className = "field";
    const title = document.createElement("span");
    title.textContent = spec.label;
    wrap.appendChild(title);
    if (spec.kind === "binary") {
      const sel = document.createElement("select");
      sel.name = spec.name;
      sel.innerHTML =
        '<option value="0">No</option><option value="1">Yes</option>';
      sel.value = String(spec.default);
      wrap.appendChild(sel);
    } else {
      const input = document.createElement("input");
      input.type = "number";
      input.name = spec.name;
      input.min = spec.min;
      input.max = spec.max;
      input.step = spec.step;
      input.value = spec.default;
      input.required = true;
      wrap.appendChild(input);
    }
    if (spec.hint) {
      const hint = document.createElement("span");
      hint.className = "hint";
      hint.textContent = spec.hint;
      wrap.appendChild(hint);
    }
    return wrap;
  }

  function fill(values) {
    for (const [k, v] of Object.entries(values)) {
      const el = form.elements.namedItem(k);
      if (el) el.value = String(v);
    }
  }

  function renderResult(data) {
    result.className = "result " + (data.probability >= 0.5 ? "higher" : "lower");
    result.innerHTML =
      '<p class="score">' +
      (data.probability * 100).toFixed(1) +
      "%</p>" +
      '<p class="label">' +
      data.label +
      "</p>" +
      "<p>" +
      data.disclaimer +
      "</p>";
  }

  function renderError(msg) {
    result.className = "result";
    result.innerHTML = "<p>" + msg + "</p>";
  }

  async function loadSchema() {
    const res = await fetch("/api/schema");
    if (!res.ok) throw new Error("Could not load the demo schema.");
    schema = await res.json();
    schema.features.forEach((spec) => form.appendChild(fieldInput(spec)));
    schema.samples.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.title;
      sampleSelect.appendChild(opt);
    });
    const m = schema.metrics;
    metricsEl.innerHTML =
      "Synthetic hold-out accuracy <strong>" +
      m.accuracy +
      "</strong>, ROC AUC <strong>" +
      m.roc_auc +
      "</strong> (n_test = " +
      m.n_test +
      "). Metrics describe invented data only.";
  }

  sampleSelect.addEventListener("change", function () {
    if (!schema || !sampleSelect.value) return;
    const sample = schema.samples.find((s) => s.id === sampleSelect.value);
    if (sample) fill(sample);
  });

  form.addEventListener("submit", async function (ev) {
    ev.preventDefault();
    const payload = {};
    for (const spec of schema.features) {
      const el = form.elements.namedItem(spec.name);
      payload[spec.name] = Number(el.value);
    }
    result.className = "result";
    result.innerHTML = '<p class="muted">Running the demo model.</p>';
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        renderError(data.detail || "The demo model could not run.");
        return;
      }
      renderResult(data);
    } catch (err) {
      renderError("The demo model could not run.");
    }
  });

  loadSchema().catch((err) => renderError(err.message));
})();
