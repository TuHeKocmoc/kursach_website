const state = {
  symbol: "BTC-USD",
  history: [],
  priceChart: null,
  forecastChart: null
};

function el(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const node = el(id);
  if (node) node.textContent = text;
}

function formatPrice(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

const LOCALE = "ru-RU";

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
  return candles.map(c => formatDate(c.time));
}

function makeCloseSeries(candles) {
  return candles.map(c => Number(c.close));
}

function makeForecastSeries(points) {
  return points.map(p => Number(p.value));
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
          tension: 0.2,
          pointRadius: 0
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
            pointStyleWidth: 36,
            generateLabels(chart) {
              const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
              for (const l of labels) {
                l.lineCap = "round";
                l.lineJoin = "round";
              }
              return labels;
            }
          }
        },
        tooltip: {
          mode: "index",
          intersect: false,
          displayColors: false
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
              const interval = document.getElementById("historyInterval")?.value || "1d";
              const step = interval === "1h" ? 4 : 2;
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
          label: "Forecast",
          data: [],
          borderColor: "#24d18f",
          borderWidth: 3,
          backgroundColor: "rgba(36, 209, 143, 0.12)",
          fill: true,
          tension: 0.2,
          pointRadius: 2
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
            pointStyleWidth: 36,
            generateLabels(chart) {
              const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
              for (const l of labels) {
                l.lineCap = "round";
                l.lineJoin = "round";
              }
              return labels;
            }
          }
        },
        tooltip: {
          mode: "index",
          intersect: false,
          displayColors: false
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

async function loadLastPrice() {
  try {
    const data = await fetchJson(`/api/market/last?symbol=${encodeURIComponent(state.symbol)}`);
    setText("symbolValue", data.symbol || state.symbol);
    setText("lastPriceValue", formatPrice(data.price));
    setText("lastTimeValue", formatDateTime(data.time, true));
  } catch (e) {
    setText("lastPriceValue", "—");
    setText("lastTimeValue", "—");
  }
}

function updatePriceChart(candles, interval) {
  const labels = makeLabels(candles);
  const series = makeCloseSeries(candles);

  if (state.priceChart?.options?.scales?.x?.ticks?.maxTicksLimit !== undefined) {
    delete state.priceChart.options.scales.x.ticks.maxTicksLimit;
  }

  state.priceChart.options.scales.x.ticks.callback = function (value, index) {
    if (interval !== "1h" && index % 2 !== 0) return "";
    return this.getLabelForValue(value);
  };

  state.priceChart.data.labels = labels;
  state.priceChart.data.datasets[0].data = series;
  state.priceChart.update();

  if (interval === "1h") {
    const current = state.priceChart.scales?.x?.ticks?.length || 0;
    const target = Math.max(2, Math.ceil(current / 2));
    state.priceChart.options.scales.x.ticks.maxTicksLimit = target;
    state.priceChart.update();
  }
}

async function loadHistory() {
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

function updateForecastChart(points) {
  const labels = points.map(p => formatDateTimeNoSeconds(p.time));
  const series = makeForecastSeries(points);
  state.forecastChart.data.labels = labels;
  state.forecastChart.data.datasets[0].data = series;
  state.forecastChart.update();
}

async function runPredict() {
  const model = el("modelSelect").value;
  const horizon = Number(el("horizonSelect").value);
  setText("predictStatus", "Running prediction...");
  setText("predGenAt", "—");
  setText("predCount", "—");
  updateForecastChart([]);
  try {
    const payload = {
      symbol: state.symbol,
      model: model,
      horizon_days: horizon,
      interval: "1d"
    };
    const data = await fetchJson(`/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const points = Array.isArray(data.forecast) ? data.forecast : [];
    setText("predGenAt", formatDateTime(data.generated_at));
    setText("predCount", String(points.length));
    updateForecastChart(points);
    setText("predictStatus", points.length ? "Done" : "No forecast points");
  } catch (e) {
    setText("predictStatus", `Error: ${e.message}`);
  }
}

function bindEvents() {
  const predictBtn = el("predictBtn");
  if (predictBtn) {
    predictBtn.addEventListener("click", async () => {
      await runPredict();
    });
  }

  const historyInterval = el("historyInterval");
  if (historyInterval) {
    historyInterval.addEventListener("change", async () => {
      await loadHistory();
    });
  }

  const historyPeriod = el("historyPeriod");
  if (historyPeriod) {
    historyPeriod.addEventListener("change", async () => {
      await loadHistory();
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  bindEvents();
  await loadLastPrice();
  await loadHistory();
});