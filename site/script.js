/* ==========================================================================
   Fooling a CIFAR-10 classifier: page behaviour
   Plain JavaScript, no libraries. All data comes from ./data/*.json, which
   the export cell at the end of p2_code.ipynb writes. Every section copes
   with a missing or placeholder file by leaving its "—" defaults in place.
   ========================================================================== */
"use strict";

const DATA_DIR = "data/";
const INTRO_EXAMPLE = "cat_7";      // which attack example section 03 shows
const FRAME_MS = 450;               // animation speed (ms per frame)
const SVG_NS = "http://www.w3.org/2000/svg";

/* ---------- small helpers ---------- */

async function loadJSON(name) {
  try {
    const res = await fetch(DATA_DIR + name, { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status + " " + res.statusText);
    return await res.json();
  } catch (err) {
    console.warn(`Could not load ${DATA_DIR}${name}:`, err.message);
    return null;
  }
}

/** Create an element: el("p", { class: "x", text: "hi" }, [children]) */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null) continue;
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

const isNum = (v) => typeof v === "number" && isFinite(v);
const fmt = {
  pct: (v) => (isNum(v) ? `${(v * 100).toFixed(v === 1 || v === 0 ? 0 : 1)}%` : "—"),
  dec: (v, d) => (isNum(v) ? v.toFixed(d) : "—"),
  int: (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "—"),
};

/** An upscaled pixel image */
function pxImage(src, alt) {
  return el("img", { class: "px", src, alt, width: 160, height: 160, loading: "lazy", decoding: "async" });
}

/** "model: frog", green when it matches the true label, red otherwise */
function labelSpan(predicted, truth, prefix = "model: ") {
  return el("span", { class: predicted === truth ? "ok" : "bad", text: prefix + predicted });
}

/* ---------- 01 key figures ---------- */

function renderKeyFigures(results) {
  const f = results && results.final;
  if (!f || f.placeholder) return;
  const set = (key, text) => { const node = document.querySelector(`#key-figures [data-key="${key}"]`); if (node) node.textContent = text; };
  set("success", fmt.pct(f.success_rate));
  set("l2", fmt.dec(f.avg_L2, 3));
  set("ssim", fmt.dec(f.avg_SSIM, 4));
  set("queries", fmt.int(f.avg_queries));
}

/* ---------- 02 model grid ---------- */

function renderModelGrid(demo, results) {
  const count = (demo && demo.n_misclassified_clean) ?? (results && results.n_misclassified_clean);
  document.querySelectorAll('[data-bind="misclassified"]').forEach((n) => {
    n.textContent = isNum(count) ? String(count) : "—";
  });

  const grid = document.getElementById("model-grid");
  if (!demo || !Array.isArray(demo.images) || !demo.images.length) {
    document.querySelector('[data-empty="model-grid"]').hidden = false;
    return;
  }
  for (const item of demo.images) {
    grid.append(el("figure", {}, [
      pxImage(item.file, `${item.true_label} photo`),
      el("figcaption", { title: `true: ${item.true_label}` }, [labelSpan(item.predicted_label, item.true_label, "")]),
    ]));
  }
}

/* ---------- 03 clean + noise = adversarial ---------- */

function renderIntro(examples) {
  const box = document.getElementById("intro-example");
  const list = (examples && examples.examples || []).filter((e) => !e.placeholder);
  const ex = list.find((e) => e.name === INTRO_EXAMPLE) || list[0];
  if (!ex) { box.append(el("p", { class: "note", text: "This example appears once data/attack_examples.json is exported." })); return; }

  const part = (src, alt, caption, label) => el("figure", {}, [
    pxImage(src, alt),
    el("figcaption", {}, [document.createTextNode(caption), label ? el("span", { class: "label" }, [label]) : null]),
  ]);
  box.append(
    part(ex.files.clean, `Original ${ex.true_label} photo`, "original photo", labelSpan(ex.pred_clean, ex.true_label)),
    el("span", { class: "op", "aria-hidden": "true", text: "+" }),
    part(ex.files.noise, "The change, magnified ten times", "tiny change (shown ×10)", el("span", { class: "cap-muted", text: `L2 = ${fmt.dec(ex.L2, 2)}` })),
    el("span", { class: "op", "aria-hidden": "true", text: "=" }),
    part(ex.files.adversarial, "Adversarial photo", "adversarial photo", labelSpan(ex.pred_adversarial, ex.true_label)),
  );
}

/* ---------- 07 results tables ---------- */

// Column definitions: key in results.json, formatter, and whether higher is better
const RESULT_COLUMNS = [
  { key: "success_rate", show: fmt.pct, better: "high" },
  { key: "avg_L2", show: (v) => fmt.dec(v, 3), better: "low" },
  { key: "avg_Linf", show: (v) => fmt.dec(v, 3), better: "low" },
  { key: "avg_SSIM", show: (v) => fmt.dec(v, 4), better: "high" },
  { key: "avg_PSNR", show: (v) => fmt.dec(v, 1), better: "high" },
  { key: "avg_queries", show: fmt.int, better: "low" },
];

function renderResults(results) {
  const tbody = document.querySelector("#results-table tbody");
  const switcher = document.getElementById("budget-switch");
  if (!results || !Array.isArray(results.methods)) {
    tbody.append(el("tr", {}, [el("td", { colspan: 8, class: "muted", text: "Results appear once data/results.json is exported." })]));
    return;
  }
  const budgets = results.budgets || [...new Set(results.methods.map((m) => m.budget).filter(isNum))];
  let current = budgets[budgets.length - 1];

  const draw = () => {
    tbody.textContent = "";
    const baseline = results.methods.filter((m) => m.budget === null || m.budget === undefined);
    const attacks = results.methods.filter((m) => m.budget === current && !/^final/i.test(m.method));
    const final = results.methods.filter((m) => /^final/i.test(m.method));
    const ranked = [...attacks, ...final];   // the baseline is a starting point, so it is not ranked

    const best = {};
    for (const col of RESULT_COLUMNS) {
      const values = ranked.map((m) => m[col.key]).filter(isNum);
      if (values.length) best[col.key] = col.better === "high" ? Math.max(...values) : Math.min(...values);
    }
    for (const m of [...baseline, ...attacks, ...final]) {
      const isFinal = /^final/i.test(m.method);
      const cells = [
        el("td", { text: m.method + (m.placeholder ? " (placeholder)" : "") }),
        el("td", { class: "num", text: isNum(m.n_images) ? String(m.n_images) : "—" }),
      ];
      for (const col of RESULT_COLUMNS) {
        const v = m[col.key];
        const text = col.show(v);
        const isBest = ranked.includes(m) && isNum(v) && v === best[col.key];
        cells.push(el("td", { class: "num" }, [isBest ? el("b", { text }) : document.createTextNode(text)]));
      }
      tbody.append(el("tr", { class: isFinal ? "final" : "" }, cells));
    }
  };

  switcher.append(el("span", { text: "Queries per image:" }));
  for (const b of budgets) {
    const btn = el("button", { type: "button", "aria-pressed": String(b === current), text: fmt.int(b) });
    btn.addEventListener("click", () => {
      current = b;
      switcher.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
      draw();
    });
    switcher.append(btn);
  }
  draw();
}

function renderImprovements(results) {
  const tbody = document.querySelector("#improvements-table tbody");
  const rows = (results && results.improvements) || [];
  if (!rows.length) {
    tbody.append(el("tr", {}, [el("td", { colspan: 5, class: "muted", text: "Appears once data/results.json is exported." })]));
    return;
  }
  for (const r of rows) {
    tbody.append(el("tr", {}, [
      el("td", { text: r.description || r.step }),
      el("td", { class: "num", text: fmt.dec(r.avg_L2, 3) }),
      el("td", { class: "num", text: fmt.dec(r.avg_SSIM, 4) }),
      el("td", { class: "num", text: isNum(r.jitter) ? `${Math.round(r.jitter * 100)}%` : "—" }),
      el("td", { class: r.kept ? "" : "muted", text: r.kept ? "yes" : "no" }),
    ]));
  }
}

/* ---------- 08 before / after pairs ---------- */

function renderPairs(examples) {
  const box = document.getElementById("pairs");
  const list = (examples && examples.examples) || [];
  if (!list.length) { box.append(el("p", { class: "note", text: "Examples appear once data/attack_examples.json is exported." })); return; }

  for (const ex of list) {
    if (ex.placeholder) {
      box.append(el("div", { class: "pair" }, [el("p", { class: "note", text: `${ex.name}: not exported yet.` })]));
      continue;
    }
    const advImg = pxImage(ex.files.adversarial, `Adversarial ${ex.true_label} photo. Press and hold to show the original.`);
    const advCap = el("figcaption", {}, [labelSpan(ex.pred_adversarial, ex.true_label)]);
    const flip = el("button", { type: "button", class: "flip", "aria-label": `Adversarial ${ex.name}. Press and hold to compare with the original.` }, [advImg]);

    // Press-and-hold swaps the adversarial image for the clean one
    const show = (clean) => {
      advImg.src = clean ? ex.files.clean : ex.files.adversarial;
      advCap.textContent = "";
      advCap.append(clean ? labelSpan(ex.pred_clean, ex.true_label, "original · model: ") : labelSpan(ex.pred_adversarial, ex.true_label));
    };
    flip.addEventListener("pointerdown", (e) => { e.preventDefault(); show(true); });
    ["pointerup", "pointerleave", "pointercancel"].forEach((t) => flip.addEventListener(t, () => show(false)));
    flip.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); show(true); } });
    flip.addEventListener("keyup", () => show(false));
    flip.addEventListener("blur", () => show(false));
    flip.addEventListener("contextmenu", (e) => e.preventDefault());

    box.append(el("div", { class: "pair" }, [
      el("figure", {}, [pxImage(ex.files.clean, `Original ${ex.true_label} photo`), el("figcaption", {}, [labelSpan(ex.pred_clean, ex.true_label)])]),
      el("figure", {}, [pxImage(ex.files.noise, "Change, magnified ten times"), el("figcaption", { class: "cap-muted", text: "change ×10" })]),
      el("figure", {}, [flip, advCap]),
      el("div", { class: "pair-stats" }, [
        el("p", { class: "name", text: ex.name }),
        el("p", { text: `L2 ${fmt.dec(ex.L2, 3)}` }),
        el("p", { text: `SSIM ${fmt.dec(ex.SSIM, 4)}` }),
        el("p", { text: `queries ${fmt.int(ex.queries)}` }),
      ]),
    ]));
  }
}

/* ---------- 06 attack player ---------- */

function setupPlayer(framesData) {
  const ui = {
    tabs: document.getElementById("player-tabs"),
    image: document.getElementById("pl-image"),
    noise: document.getElementById("pl-noise"),
    clean: document.getElementById("pl-clean"),
    label: document.getElementById("pl-label"),
    truth: document.getElementById("pl-true"),
    play: document.getElementById("pl-play"),
    slider: document.getElementById("pl-slider"),
    count: document.getElementById("pl-count"),
    stage: document.getElementById("pl-stage"),
    l2: document.getElementById("pl-l2"),
    queries: document.getElementById("pl-queries"),
    chart: document.getElementById("pl-chart"),
  };
  const items = (framesData && framesData.images || []).filter((x) => Array.isArray(x.frames) && x.frames.length);
  if (!items.length) {
    document.getElementById("player-empty").hidden = false;
    document.querySelector("#player .player-panels").hidden = true;
    document.querySelector("#player .player-controls").hidden = true;
    return;
  }

  let item = items[0], index = 0, timer = null, chart = null;

  // Preload every frame so playback doesn't flicker
  for (const it of items) for (const f of it.frames) { new Image().src = f.file; new Image().src = f.noise; }

  const stop = () => { clearInterval(timer); timer = null; ui.play.textContent = index >= item.frames.length - 1 ? "Replay" : "Play"; };
  const show = (i) => {
    index = Math.max(0, Math.min(item.frames.length - 1, i));
    const f = item.frames[index];
    ui.image.src = f.file;
    ui.noise.src = f.noise;
    ui.label.textContent = f.label;
    ui.label.className = "pl-answer-value mono " + (f.label === item.true_label ? "ok" : "bad");
    ui.slider.value = index;
    ui.count.textContent = `${index + 1} / ${item.frames.length}`;
    ui.stage.textContent = f.stage;
    ui.l2.textContent = fmt.dec(f.l2, 3);
    ui.queries.textContent = fmt.int(f.queries);
    if (chart) chart.moveTo(index);
  };
  const select = (it) => {
    stop();
    item = it;
    ui.tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.name === it.name)));
    ui.clean.src = it.clean;
    ui.truth.textContent = `true answer: ${it.true_label}`;
    ui.slider.max = it.frames.length - 1;
    chart = drawChart(ui.chart, it.frames);
    show(0);
    ui.play.textContent = "Play";
  };

  for (const it of items) {
    const b = el("button", { type: "button", role: "tab", "data-name": it.name, "aria-selected": "false", text: it.name });
    b.addEventListener("click", () => select(it));
    ui.tabs.append(b);
  }
  ui.play.addEventListener("click", () => {
    if (timer) return stop();
    if (index >= item.frames.length - 1) show(0);
    ui.play.textContent = "Pause";
    timer = setInterval(() => {
      if (index >= item.frames.length - 1) return stop();
      show(index + 1);
    }, FRAME_MS);
  });
  ui.slider.addEventListener("input", () => { stop(); show(Number(ui.slider.value)); });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { chart = drawChart(ui.chart, item.frames); chart.moveTo(index); }, 150);
  });
  select(items[0]);
}

/** L2 (log scale) against queries. Returns { moveTo(i) } to move the dot. */
function drawChart(svgNode, frames) {
  svgNode.textContent = "";
  // Draw at the element's real width so text stays at its true size on any screen
  const W = Math.max(300, Math.round(svgNode.getBoundingClientRect().width) || 640), H = 230, m = { l: 48, r: 16, t: 28, b: 34 };
  svgNode.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const qs = frames.map((f) => f.queries), ds = frames.map((f) => Math.max(f.l2, 1e-3));
  const qMax = Math.max(...qs, 1);
  const lo = Math.floor(Math.log10(Math.min(...ds))), hi = Math.ceil(Math.log10(Math.max(...ds)));
  const x = (q) => m.l + (q / qMax) * (W - m.l - m.r);
  const y = (d) => m.t + (1 - (Math.log10(d) - lo) / Math.max(hi - lo, 1)) * (H - m.t - m.b);

  // grid + y labels at 1, 3 x 10^k
  for (let k = lo; k <= hi; k++) {
    for (const mult of [1, 3]) {
      const v = mult * 10 ** k;
      if (Math.log10(v) > hi + 1e-9) continue;
      svgNode.append(svg("line", { class: "grid", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      const t = svg("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" });
      t.textContent = v >= 1 ? String(v) : v.toFixed(Math.max(0, -k));
      svgNode.append(t);
    }
  }
  // x axis + labels
  svgNode.append(svg("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  const step = qMax > 2000 ? (W < 480 ? 2000 : 1000) : 500;
  for (let q = 0; q <= qMax; q += step) {
    const t = svg("text", { x: x(q), y: H - m.b + 16, "text-anchor": "middle" });
    t.textContent = q.toLocaleString("en-US");
    svgNode.append(t);
  }
  const xl = svg("text", { x: W - m.r, y: H - 4, "text-anchor": "end" }); xl.textContent = "queries used"; svgNode.append(xl);
  const yl = svg("text", { x: m.l - 40, y: 12 }); yl.textContent = "L2 distance (log scale)"; svgNode.append(yl);

  // full path (light) and the part already played (accent)
  const pts = frames.map((f, i) => `${x(qs[i]).toFixed(1)},${y(ds[i]).toFixed(1)}`);
  svgNode.append(svg("polyline", { class: "line", points: pts.join(" "), opacity: "0.25" }));
  const done = svg("polyline", { class: "line-done", points: pts[0] });
  const dot = svg("circle", { class: "dot", r: 5, cx: x(qs[0]), cy: y(ds[0]) });
  svgNode.append(done, dot);

  return {
    moveTo(i) {
      done.setAttribute("points", pts.slice(0, i + 1).join(" "));
      dot.setAttribute("cx", x(qs[i]));
      dot.setAttribute("cy", y(ds[i]));
    },
  };
}

/* ---------- fade-in on scroll ---------- */

function setupReveal() {
  const nodes = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { nodes.forEach((n) => n.classList.add("visible")); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("visible"); io.unobserve(e.target); }
  }, { rootMargin: "0px 0px -10% 0px" });
  nodes.forEach((n) => io.observe(n));
}

/* ---------- start ---------- */

(async function init() {
  setupReveal();
  if (location.protocol === "file:") {
    // Browsers block fetch() for local files, so tell the reader how to serve the folder
    const notice = el("p", { class: "note file-notice",
      text: "Opened as a local file, so the data can't load. In the site folder run: python -m http.server, then open http://localhost:8000" });
    document.querySelector("main").prepend(notice);
  }
  const [results, demo, examples, frames] = await Promise.all([
    loadJSON("results.json"), loadJSON("model_demo.json"), loadJSON("attack_examples.json"), loadJSON("frames.json"),
  ]);
  renderKeyFigures(results);
  renderModelGrid(demo, results);
  renderIntro(examples);
  setupPlayer(frames);
  renderResults(results);
  renderImprovements(results);
  renderPairs(examples);
})();
