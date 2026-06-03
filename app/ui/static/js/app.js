const state = {
  symbol: "BTC-USD",
  history: [],
  lastPrediction: null,
  priceChart: null,
  forecastChart: null
};

const LOCALE = "en-US";

function el(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const node = el(id);
  if (node) node.textContent = text;
}

function formatNumber(x, digits = 2) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatPrice(x) {
  return formatNumber(x, 2);
}

function formatPercent(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return `${n.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}

function formatDate(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(LOCALE, { year: "numeric", month: "2-digit", day: "2-digit" });
}

function formatDateTime(iso, withSeconds = true) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const opts = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  };
  if (withSeconds) opts.second = "2-digit";
  return d.toLocaleString(LOCALE, opts);
}

function formatDateTimeNoSeconds(iso) {
  return formatDateTime(iso, false);
}

function makeLabels(candles) {
  const interval = el("historyInterval")?.value || "1d";
  return candles.map(c => interval === "1h" ? formatDateTimeNoSeconds(c.time) : formatDate(c.time));
}

function makeSeries(candles, field) {
  return candles.map(c => {
    const value = c[field];
    if (value === null || value === undefined) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  });
}

function setDatasetVisibility() {
  if (!state.priceChart) return;
  const visible = Boolean(el("indicatorToggle")?.checked);
  state.priceChart.data.datasets.forEach((dataset, index) => {
    if (index > 0) dataset.hidden = !visible;
  });
  state.priceChart.update();
}

function initCharts() {
  const priceCtx = el("priceChart");
  const forecastCtx = el("forecastChart");

  state.priceChart = new Chart(priceCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Close",
          data: [],
          borderColor: "#2a4cff",
          borderWidth: 3,
          backgroundColor: "rgba(42, 76, 255, 0.15)",
          fill: true,
          tension: 0.18,
          pointRadius: 0,
          spanGaps: true
        },
        {
          label: "Tenkan-sen",
          data: [],
          borderColor: "#ffb84d",
          borderWidth: 1.8,
          fill: false,
          tension: 0.16,
          pointRadius: 0,
          spanGaps: true
        },
        {
          label: "Kijun-sen",
          data: [],
          borderColor: "#ff5c7a",
          borderWidth: 1.8,
          fill: false,
          tension: 0.16,
          pointRadius: 0,
          spanGaps: true
        },
        {
          label: "Senkou A",
          data: [],
          borderColor: "#24d18f",
          borderWidth: 1.5,
          fill: false,
          tension: 0.16,
          pointRadius: 0,
          spanGaps: true
        },
        {
          label: "Senkou B",
          data: [],
          borderColor: "#b37cff",
          borderWidth: 1.5,
          fill: false,
          tension: 0.16,
          pointRadius: 0,
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#e9eefc",
            usePointStyle: true,
            pointStyle: "line",
            pointStyleWidth: 36
          }
        },
        tooltip: {
          mode: "index",
          intersect: false,
          displayColors: true
        }
      },
      interaction: {
        mode: "index",
        intersect: false
      },
      scales: {
        x: {
          ticks: {
            color: "#9bb0d1",
            maxRotation: 0,
            autoSkip: true,
            callback: function (value, index) {
              const interval = el("historyInterval")?.value || "1d";
              const step = interval === "1h" ? 6 : 2;
              if (index % step !== 0) return "";
              return this.getLabelForValue(value);
            }
          },
          grid: { color: "rgba(255,255,255,0.06)" }
        },
        y: {
          ticks: { color: "#9bb0d1" },
          grid: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });

  state.forecastChart = new Chart(forecastCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Last price",
          data: [],
          borderColor: "#8ea4d2",
          borderWidth: 2,
          backgroundColor: "rgba(142, 164, 210, 0.08)",
          fill: false,
          tension: 0.1,
          pointRadius: 3,
          spanGaps: true
        },
        {
          label: "Forecast",
          data: [],
          borderColor: "#24d18f",
          borderWidth: 3,
          backgroundColor: "rgba(36, 209, 143, 0.12)",
          fill: true,
          tension: 0.2,
          pointRadius: 3,
          spanGaps: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#e9eefc",
            usePointStyle: true,
            pointStyle: "line",
            pointStyleWidth: 36
          }
        },
        tooltip: {
          mode: "index",
          intersect: false,
          displayColors: true
        }
      },
      interaction: {
        mode: "index",
        intersect: false
      },
      scales: {
        x: {
          ticks: {
            color: "#9bb0d1",
            maxRotation: 0,
            autoSkip: true,
            callback: function (value, index) {
              if (index % 2 !== 0) return "";
              return this.getLabelForValue(value);
            }
          },
          grid: { color: "rgba(255,255,255,0.06)" }
        },
        y: {
          ticks: { color: "#9bb0d1" },
          grid: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) {
    const msg = data && data.detail ? String(data.detail) : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function readSymbol() {
  const value = (el("symbolInput")?.value || "BTC-USD").trim().toUpperCase();
  state.symbol = value || "BTC-USD";
  if (el("symbolInput")) el("symbolInput").value = state.symbol;
  return state.symbol;
}

async function loadLastPrice() {
  readSymbol();
  try {
    const data = await fetchJson(`/api/market/last?symbol=${encodeURIComponent(state.symbol)}`);
    setText("symbolValue", data.symbol || state.symbol);
    setText("lastPriceValue", formatPrice(data.price));
    setText("lastTimeValue", formatDateTime(data.time, true));
  } catch (e) {
    setText("symbolValue", state.symbol);
    setText("lastPriceValue", "—");
    setText("lastTimeValue", "—");
  }
}

function updatePriceChart(candles, interval) {
  const labels = makeLabels(candles);
  const datasets = state.priceChart.data.datasets;
  datasets[0].data = makeSeries(candles, "close");
  datasets[1].data = makeSeries(candles, "tenkan_sen");
  datasets[2].data = makeSeries(candles, "kijun_sen");
  datasets[3].data = makeSeries(candles, "senkou_span_a");
  datasets[4].data = makeSeries(candles, "senkou_span_b");

  state.priceChart.data.labels = labels;
  state.priceChart.options.scales.x.ticks.callback = function (value, index) {
    const step = interval === "1h" ? 6 : 2;
    if (index % step !== 0) return "";
    return this.getLabelForValue(value);
  };
  state.priceChart.update();
  setDatasetVisibility();
}

async function loadHistory() {
  readSymbol();
  const period = el("historyPeriod").value;
  const interval = el("historyInterval").value;
  setText("historyStatus", "Loading history...");
  try {
    const url = `/api/market/history?symbol=${encodeURIComponent(state.symbol)}&interval=${encodeURIComponent(interval)}&period=${encodeURIComponent(period)}`;
    const data = await fetchJson(url);
    state.history = Array.isArray(data.candles) ? data.candles : [];
    if (!state.history.length) {
      setText("historyStatus", "No data");
      updatePriceChart([], interval);
      return;
    }
    updatePriceChart(state.history, interval);
    const start = state.history[0].time;
    const end = state.history[state.history.length - 1].time;
    const fmt = interval === "1h" ? formatDateTimeNoSeconds : formatDate;
    setText("historyStatus", `${state.history.length} points, ${fmt(start)} → ${fmt(end)}`);
  } catch (e) {
    setText("historyStatus", `Error: ${e.message}`);
    updatePriceChart([], interval);
  }
}

function resetMetrics() {
  setText("metricRmse", "—");
  setText("metricMae", "—");
  setText("metricDirAcc", "—");
  setText("metricNaiveMae", "—");
}

function updateMetrics(metrics) {
  if (!metrics) {
    resetMetrics();
    return;
  }
  setText("metricRmse", formatPrice(metrics.rmse));
  setText("metricMae", formatPrice(metrics.mae));
  setText("metricDirAcc", formatPercent(metrics.directional_accuracy));
  setText("metricNaiveMae", formatPrice(metrics.naive_mae));
}

function updateForecastChart(data) {
  const points = Array.isArray(data?.forecast) ? data.forecast : [];
  if (!points.length) {
    state.forecastChart.data.labels = [];
    state.forecastChart.data.datasets[0].data = [];
    state.forecastChart.data.datasets[1].data = [];
    state.forecastChart.update();
    return;
  }

  const labels = ["Last", ...points.map(p => formatDate(p.time))];
  const lastPrice = Number(data.last_price);
  const forecastSeries = [Number.isFinite(lastPrice) ? lastPrice : null, ...points.map(p => Number(p.value))];
  const lastSeries = [Number.isFinite(lastPrice) ? lastPrice : null, ...points.map(() => null)];
  state.forecastChart.data.labels = labels;
  state.forecastChart.data.datasets[0].data = lastSeries;
  state.forecastChart.data.datasets[1].data = forecastSeries;
  state.forecastChart.update();
}

function showWarnings(warnings) {
  const box = el("warningBox");
  if (!box) return;
  if (!warnings || !warnings.length) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  box.hidden = false;
  box.textContent = warnings.join(" ");
}

async function runPredict() {
  readSymbol();
  const model = el("modelSelect").value;
  const horizon = Number(el("horizonSelect").value);
  setText("predictStatus", "Running prediction...");
  setText("predGenAt", "—");
  setText("predCount", "—");
  setText("predLastValue", "—");
  resetMetrics();
  showWarnings([]);
  updateForecastChart(null);

  try {
    const payload = {
      symbol: state.symbol,
      model: model,
      horizon_days: horizon,
      interval: "1d"
    };
    const data = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const points = Array.isArray(data.forecast) ? data.forecast : [];
    state.lastPrediction = data;
    setText("predGenAt", formatDateTime(data.generated_at));
    setText("predCount", String(points.length));
    setText("predLastValue", points.length ? formatPrice(points[points.length - 1].value) : "—");
    updateForecastChart(data);
    updateMetrics(data.metrics);
    showWarnings(data.warnings || []);
    const metricNote = data.metrics ? ` Backtest horizon: ${data.metrics.horizon_days}d.` : "";
    setText("predictStatus", points.length ? `Done.${metricNote}` : "No forecast points");
  } catch (e) {
    setText("predictStatus", `Error: ${e.message}`);
  }
}

function clearMetricsTable(message) {
  const body = el("metricsTableBody");
  body.innerHTML = "";
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 9;
  cell.className = "empty-table";
  cell.textContent = message;
  row.appendChild(cell);
  body.appendChild(row);
}

function renderMetricsTable(metrics) {
  const body = el("metricsTableBody");
  body.innerHTML = "";
  if (!metrics || !metrics.length) {
    clearMetricsTable("No metrics available.");
    return;
  }
  const names = {
    naive: "Naive",
    pdt: "PDT",
    xgb: "XGBoost",
    lstm: "LSTM"
  };
  for (const rowData of metrics) {
    const row = document.createElement("tr");
    const values = [
      names[rowData.model] || rowData.model,
      `${rowData.horizon_days}d`,
      rowData.train_size,
      rowData.test_size,
      formatPrice(rowData.rmse),
      formatPrice(rowData.mae),
      formatPercent(rowData.mape),
      formatPercent(rowData.directional_accuracy),
      formatPrice(rowData.naive_rmse)
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

async function runEvaluation() {
  readSymbol();
  const horizon = Number(el("horizonSelect").value || 1);
  const evalHorizon = Math.min(30, Math.max(1, horizon));
  setText("evaluateStatus", "Evaluating models...");
  clearMetricsTable("Running...");
  try {
    const url = `/api/evaluate?symbol=${encodeURIComponent(state.symbol)}&horizon_days=${encodeURIComponent(evalHorizon)}`;
    const data = await fetchJson(url);
    renderMetricsTable(data.metrics || []);
    setText("evaluateStatus", `Done at ${formatDateTime(data.generated_at)}.`);
  } catch (e) {
    clearMetricsTable("Evaluation failed.");
    setText("evaluateStatus", `Error: ${e.message}`);
  }
}

function bindEvents() {
  el("predictBtn")?.addEventListener("click", runPredict);
  el("evaluateBtn")?.addEventListener("click", runEvaluation);
  el("reloadBtn")?.addEventListener("click", async () => {
    readSymbol();
    await loadLastPrice();
    await loadHistory();
  });
  el("historyInterval")?.addEventListener("change", loadHistory);
  el("historyPeriod")?.addEventListener("change", loadHistory);
  el("indicatorToggle")?.addEventListener("change", setDatasetVisibility);
  el("symbolInput")?.addEventListener("keydown", async event => {
    if (event.key === "Enter") {
      readSymbol();
      await loadLastPrice();
      await loadHistory();
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  bindEvents();
  await loadLastPrice();
  await loadHistory();
});