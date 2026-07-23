/*
  MotherTongueIndex web app.
  Works two ways:
   - Live: POSTs to /api/analyze (python web/server.py) to tokenize any pasted
     text with the real tokenizers.
   - Static: with no server, the Results table and the sample languages use
     precomputed real data baked into assets/tables.js.
*/
(function () {
  "use strict";

  var TABLES = window.MTI_TABLES || { models: [], by_language: {} };
  var SAMPLES = window.MTI_SAMPLES || [];

  // Models shown as selectable chips. The first four have baked offline data;
  // all work when the live server is running.
  var CHIP_MODELS = [
    { id: "gpt-4o", label: "GPT-4o" },
    { id: "gpt-4", label: "GPT-4" },
    { id: "claude", label: "Claude", est: true },
    { id: "gemini", label: "Gemini", est: true },
    { id: "qwen3", label: "Qwen3", live: true },
    { id: "deepseek", label: "DeepSeek", live: true },
    { id: "sarvam1", label: "Sarvam-1", live: true },
    { id: "xlmr", label: "XLM-R", live: true }
  ];
  var selected = { "gpt-4o": true, "gpt-4": true, "claude": true, "gemini": true };

  var $ = function (s) { return document.querySelector(s); };

  // ---- nav (mobile) ----
  var navToggle = $("#navToggle");
  if (navToggle) navToggle.addEventListener("click", function () {
    var nav = $("#nav");
    var open = nav.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });

  // ---- xEN colour helper ----
  function xenClass(v) { if (v == null) return ""; if (v < 1.3) return "good"; if (v < 2.5) return "warn"; return "bad"; }
  function fmtx(v) { return v == null ? "-" : v.toFixed(2) + "x"; }

  // ---- Results table (static, from baked data) ----
  function renderResults() {
    var t = $("#resultsTable");
    if (!t || !TABLES.models.length) return;
    var models = TABLES.models;
    var head = "<tr><th>Language</th><th>Script</th>" +
      models.map(function (m) { return "<th>" + m + "</th>"; }).join("") + "</tr>";
    t.querySelector("thead").innerHTML = head;

    var rows = Object.keys(TABLES.by_language).map(function (code) {
      var row = TABLES.by_language[code];
      var cells = models.map(function (m) {
        var d = row.models[m];
        if (!d) return "<td>-</td>";
        var x = d.vs_english;
        var est = d.estimated ? '<span class="tag-est">est</span>' : "";
        return '<td><span class="xen ' + xenClass(x) + '">' + fmtx(x) + "</span>" + est + "</td>";
      }).join("");
      return "<tr><td>" + row.language + "</td><td>" + row.script + "</td>" + cells + "</tr>";
    }).join("");
    t.querySelector("tbody").innerHTML = rows;
  }

  // ---- Sample + model chips ----
  function renderSamples() {
    var box = $("#samples");
    if (!box) return;
    box.innerHTML = SAMPLES.slice(0, 12).map(function (s) {
      return '<button class="chip" data-text="' + s.text.replace(/"/g, "&quot;") +
        '">' + s.language + "</button>";
    }).join("");
    box.addEventListener("click", function (e) {
      var b = e.target.closest(".chip"); if (!b) return;
      $("#input").value = b.getAttribute("data-text");
      run();
    });
  }

  function renderModelChips() {
    var box = $("#models");
    if (!box) return;
    box.innerHTML = CHIP_MODELS.map(function (m) {
      return '<button class="chip' + (m.est ? " est" : "") + '" data-id="' + m.id +
        '" aria-pressed="' + (selected[m.id] ? "true" : "false") + '">' + m.label +
        (m.live ? " *" : "") + "</button>";
    }).join("");
    box.addEventListener("click", function (e) {
      var b = e.target.closest(".chip"); if (!b) return;
      var id = b.getAttribute("data-id");
      selected[id] = !selected[id];
      b.setAttribute("aria-pressed", String(!!selected[id]));
    });
  }
  function chosenModels() { return CHIP_MODELS.map(function (m) { return m.id; }).filter(function (id) { return selected[id]; }); }

  // ---- render a live/analysis table ----
  function renderLiveTable(results) {
    var wrap = $("#liveTableWrap"), t = $("#liveTable");
    wrap.hidden = false;
    var head = "<thead><tr><th>Model</th><th>tokens</th><th>words</th><th>fert.</th><th>xEN</th><th>b/tok</th><th></th></tr></thead>";
    var body = results.map(function (r) {
      if (!r.available) return "<tr><td>" + r.display + "</td><td colspan='6'>unavailable</td></tr>";
      var m = r.metrics;
      var est = m.estimated ? '<span class="tag-est">est</span>' : "";
      return "<tr><td>" + r.display + "</td><td>" + m.n_tokens + "</td><td>" + m.n_words +
        "</td><td>" + m.fertility.toFixed(2) + '</td><td><span class="xen ' + xenClass(r.vs_english) +
        '">' + fmtx(r.vs_english) + "</span></td><td>" + m.bytes_per_token.toFixed(1) + "</td><td>" + est + "</td></tr>";
    }).join("");
    t.innerHTML = head + "<tbody>" + body + "</tbody>";
  }

  function renderTokens(results) {
    var box = $("#tokenViz");
    var withTok = results.filter(function (r) { return r.tokens && r.tokens.length; });
    if (!withTok.length) { box.hidden = true; return; }
    box.hidden = false;
    box.innerHTML = withTok.map(function (r) {
      var toks = r.tokens.map(function (tk) {
        return '<span class="tok">' + (tk === "" ? "␣" : escapeHtml(tk).replace(/\n/g, "⏎")) + "</span>";
      }).join("");
      return "<h3>" + r.display + " (" + r.metrics.n_tokens + " tokens)</h3><div class='tokens'>" + toks + "</div>";
    }).join("");
  }
  function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  // ---- static fallback: look up a pasted sample by exact text ----
  function staticFallback(text) {
    var s = SAMPLES.find(function (x) { return x.text.trim() === text.trim(); });
    if (!s) return null;
    var row = TABLES.by_language[s.code];
    if (!row) return null;
    return TABLES.models.map(function (mid) {
      var d = row.models[mid];
      return {
        display: mid, available: !!d, vs_english: d ? d.vs_english : null, tokens: [],
        metrics: d ? { n_tokens: d.tokens, n_words: d.words, fertility: d.fertility, bytes_per_token: d.bytes_per_token, estimated: d.estimated } : null
      };
    });
  }

  var showTokens = false;
  var tBtn = $("#toggleTokens");
  if (tBtn) tBtn.addEventListener("click", function () { showTokens = !showTokens; run(); });

  function setStatus(msg) { var s = $("#status"); if (s) s.textContent = msg; }

  function run() {
    var text = $("#input").value;
    if (!text.trim()) { setStatus("Type or pick a sample first."); return; }
    var models = chosenModels();
    if (!models.length) { setStatus("Select at least one model."); return; }
    setStatus("Analyzing...");

    fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, models: models, show: showTokens })
    }).then(function (r) {
      if (!r.ok) throw new Error("no server");
      return r.json();
    }).then(function (data) {
      renderLiveTable(data.results);
      if (showTokens) renderTokens(data.results); else $("#tokenViz").hidden = true;
      setStatus("Live: real tokenizers.");
    }).catch(function () {
      var fb = staticFallback(text);
      if (fb) {
        renderLiveTable(fb);
        $("#tokenViz").hidden = true;
        setStatus("No live server: showing precomputed real data for this sample. Run python web/server.py for any text and more models.");
      } else {
        $("#liveTableWrap").hidden = true;
        setStatus("No live server running. Start it with: python web/server.py  (then any text works). Or pick a sample above.");
      }
    });
  }

  var runBtn = $("#run");
  if (runBtn) runBtn.addEventListener("click", run);
  var input = $("#input");
  if (input) input.addEventListener("keydown", function (e) { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });

  renderResults();
  renderSamples();
  renderModelChips();
})();
