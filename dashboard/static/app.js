/**
 * High-Performance Client Application for Financial Forecasting Showcase
 */

// Global state
const state = {
  overview: null,
  forecasts: [],
  stationarity: null,
  acfPacf: null,
  history: [],
  qaList: [],
  activeRange: 'all',
  charts: {},
  tablePage: 1,
  tablePageSize: 25,
  filteredHistory: []
};

// Colors matching dark fintech theme
const THEME = {
  actual: '#38bdf8', // Sky Blue
  lstm: '#10b981',   // Emerald Green
  arima: '#f59e0b',  // Amber
  residualArima: 'rgba(245, 158, 11, 0.4)',
  residualLstm: 'rgba(16, 185, 129, 0.4)',
  gridColor: '#1e293b',
  textColor: '#94a3b8'
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initActionButtons();
  loadAllData();
});

// -------------------------------------------------------------
// TAB NAVIGATION
// -------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = `tab-${tab.getAttribute('data-tab')}`;
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // Trigger redraw of charts on tab switch if needed
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
    });
  });
}

function initActionButtons() {
  // Presentation View Toggle
  const btnPres = document.getElementById('btn-toggle-presentation');
  if (btnPres) {
    btnPres.addEventListener('click', () => {
      document.body.classList.toggle('presentation-mode');
      if (document.body.classList.contains('presentation-mode')) {
        btnPres.innerHTML = '<span>❌</span> Exit Presentation';
        if (document.documentElement.requestFullscreen) {
          document.documentElement.requestFullscreen().catch(() => {});
        }
      } else {
        btnPres.innerHTML = '<span>📽️</span> Presentation View';
        if (document.exitFullscreen && document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        }
      }
    });
  }

  // Print Report
  const btnPrint = document.getElementById('btn-print-report');
  if (btnPrint) {
    btnPrint.addEventListener('click', () => {
      window.print();
    });
  }

  // Range Selector
  const rangeBtns = document.querySelectorAll('#forecast-range-selector button');
  rangeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      rangeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeRange = btn.getAttribute('data-range');
      renderForecastChart();
    });
  });

  // Reset Zoom
  const btnReset = document.getElementById('btn-reset-zoom');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      rangeBtns.forEach(b => b.classList.remove('active'));
      rangeBtns[0].classList.add('active');
      state.activeRange = 'all';
      renderForecastChart();
    });
  }

  // Checkbox Series Toggles
  ['toggle-actual', 'toggle-lstm', 'toggle-arima'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', () => renderForecastChart());
    }
  });

  // Table Search and Pagination
  const searchInput = document.getElementById('table-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) {
        state.filteredHistory = state.history;
      } else {
        state.filteredHistory = state.history.filter(item => item.date.includes(q));
      }
      state.tablePage = 1;
      renderTablePage();
    });
  }

  const btnPrev = document.getElementById('btn-table-prev');
  const btnNext = document.getElementById('btn-table-next');
  if (btnPrev && btnNext) {
    btnPrev.addEventListener('click', () => {
      if (state.tablePage > 1) {
        state.tablePage--;
        renderTablePage();
      }
    });
    btnNext.addEventListener('click', () => {
      const maxPages = Math.ceil(state.filteredHistory.length / state.tablePageSize);
      if (state.tablePage < maxPages) {
        state.tablePage++;
        renderTablePage();
      }
    });
  }

  // CSV Export
  const btnExport = document.getElementById('btn-export-csv');
  if (btnExport) {
    btnExport.addEventListener('click', exportDatasetCSV);
  }
}

// -------------------------------------------------------------
// DATA LOADING
// -------------------------------------------------------------
async function loadAllData() {
  try {
    const [overviewRes, forecastsRes, stationarityRes, acfPacfRes, historyRes, qaRes] = await Promise.all([
      fetch('/api/overview').then(r => r.json()).catch(() => null),
      fetch('/api/forecasts').then(r => r.json()).catch(() => []),
      fetch('/api/stationarity').then(r => r.json()).catch(() => null),
      fetch('/api/acf_pacf').then(r => r.json()).catch(() => null),
      fetch('/api/history').then(r => r.json()).catch(() => []),
      fetch('/api/qa').then(r => r.json()).catch(() => [])
    ]);

    state.overview = overviewRes;
    state.forecasts = forecastsRes;
    state.stationarity = stationarityRes;
    state.acfPacf = acfPacfRes;
    state.history = historyRes;
    state.filteredHistory = historyRes;
    state.qaList = qaRes;

    updateOverviewUI();
    renderForecastChart();
    renderStationarityTab();
    renderResidualsTab();
    renderVivaAccordion();
    renderTablePage();
  } catch (err) {
    console.error('Failed to load dashboard data:', err);
  }
}

// -------------------------------------------------------------
// UI UPDATES: OVERVIEW & KPIS
// -------------------------------------------------------------
function updateOverviewUI() {
  if (!state.overview || !state.overview.metrics) return;

  const m = state.overview.metrics;
  const im = m.improvements || {};
  const lstm = m.LSTM || {};
  const arima = m.ARIMA || {};

  setText('kpi-lstm-mae', `$${lstm.MAE.toFixed(2)}`);
  setText('kpi-arima-mae', `$${arima.MAE.toFixed(2)}`);
  setText('kpi-mae-delta', `-${im.MAE}%`);

  setText('kpi-lstm-rmse', `$${lstm.RMSE.toFixed(2)}`);
  setText('kpi-arima-rmse', `$${arima.RMSE.toFixed(2)}`);
  setText('kpi-rmse-delta', `-${im.RMSE}%`);

  setText('kpi-lstm-mape', `${lstm.MAPE.toFixed(2)}%`);
  setText('kpi-arima-mape', `${arima.MAPE.toFixed(2)}%`);
  setText('kpi-mape-delta', `-${im.MAPE}%`);

  if (lstm.Directional_Accuracy !== undefined) {
    setText('kpi-lstm-dir', `${lstm.Directional_Accuracy.toFixed(1)}%`);
    setText('kpi-arima-dir', `${arima.Directional_Accuracy.toFixed(1)}% (Drift)`);
  }

  // Header badges
  if (state.overview.ticker) {
    setText('header-ticker', `TICKER: ${state.overview.ticker}`);
  }
  if (state.overview.total_samples) {
    setText('header-samples', `${state.overview.total_samples.toLocaleString()} SAMPLES`);
  }
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

// -------------------------------------------------------------
// CHARTING: PRIMARY FORECAST CHART
// -------------------------------------------------------------
function renderForecastChart() {
  const canvas = document.getElementById('forecastCanvas');
  if (!canvas || !state.forecasts || state.forecasts.length === 0) return;

  // Filter by active range
  let dataSlice = state.forecasts;
  if (state.activeRange === '1y') {
    dataSlice = state.forecasts.slice(-252);
  } else if (state.activeRange === '6m') {
    dataSlice = state.forecasts.slice(-126);
  } else if (state.activeRange === '3m') {
    dataSlice = state.forecasts.slice(-63);
  }

  const labels = dataSlice.map(d => d.date);
  const actualVals = dataSlice.map(d => d.actual);
  const lstmVals = dataSlice.map(d => d.lstm);
  const arimaVals = dataSlice.map(d => d.arima);

  const showActual = document.getElementById('toggle-actual')?.checked ?? true;
  const showLstm = document.getElementById('toggle-lstm')?.checked ?? true;
  const showArima = document.getElementById('toggle-arima')?.checked ?? true;

  const datasets = [];

  if (showActual) {
    datasets.push({
      label: 'Actual Closing Price',
      data: actualVals,
      borderColor: THEME.actual,
      backgroundColor: 'rgba(56, 189, 248, 0.04)',
      borderWidth: 2.2,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.1,
      fill: false
    });
  }

  if (showLstm) {
    datasets.push({
      label: 'LSTM Forecast (Seq=60)',
      data: lstmVals,
      borderColor: THEME.lstm,
      backgroundColor: 'rgba(16, 185, 129, 0.04)',
      borderWidth: 2,
      borderDash: [5, 4],
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.1,
      fill: false
    });
  }

  if (showArima) {
    datasets.push({
      label: 'ARIMA(0, 1, 0) Forecast',
      data: arimaVals,
      borderColor: THEME.arima,
      backgroundColor: 'rgba(245, 158, 11, 0.04)',
      borderWidth: 1.8,
      borderDash: [2, 2],
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0,
      fill: false
    });
  }

  if (typeof Chart !== 'undefined') {
    if (state.charts.forecast) {
      state.charts.forecast.destroy();
    }

    state.charts.forecast = new Chart(canvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            borderColor: '#334155',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const val = ctx.parsed.y !== null ? `$${ctx.parsed.y.toFixed(2)}` : 'N/A';
                return ` ${ctx.dataset.label}: ${val}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: THEME.gridColor },
            ticks: {
              color: THEME.textColor,
              maxTicksLimit: 12,
              font: { family: 'JetBrains Mono', size: 10 }
            }
          },
          y: {
            grid: { color: THEME.gridColor },
            ticks: {
              color: THEME.textColor,
              callback: (v) => `$${v}`,
              font: { family: 'JetBrains Mono', size: 10 }
            }
          }
        }
      }
    });
  } else {
    // Graceful offline canvas fallback renderer
    renderNativeCanvasLineChart(canvas, labels, datasets);
  }
}

// -------------------------------------------------------------
// TAB 2: STATIONARITY & DIAGNOSTICS CHARTS
// -------------------------------------------------------------
function renderStationarityTab() {
  if (state.stationarity) {
    const raw = state.stationarity.raw;
    const diff = state.stationarity.differenced;

    if (raw) {
      setText('adf-raw-stat', raw.adf_statistic.toFixed(4));
      setText('adf-raw-pval', `${raw.p_value.toFixed(4)} (${raw.p_value > 0.05 ? '> 0.05' : '< 0.05'})`);
      setText('adf-raw-c1', raw.critical_values['1%'].toFixed(4));
      setText('adf-raw-c5', raw.critical_values['5%'].toFixed(4));
      setText('adf-raw-c10', raw.critical_values['10%'].toFixed(4));
      setText('adf-raw-lags', `${raw.used_lag} lags (AIC selected)`);
    }

    if (diff) {
      setText('adf-diff-stat', diff.adf_statistic.toFixed(4));
      setText('adf-diff-pval', `${diff.p_value.toExponential(2)} (<< 0.05)`);
      setText('adf-diff-c1', diff.critical_values['1%'].toFixed(4));
      setText('adf-diff-c5', diff.critical_values['5%'].toFixed(4));
      setText('adf-diff-c10', diff.critical_values['10%'].toFixed(4));
      setText('adf-diff-lags', `${diff.used_lag} lags`);
    }
  }

  // Rolling Stats Chart (sampled for snappy performance)
  const rollCanvas = document.getElementById('rollingStatsCanvas');
  if (rollCanvas && state.history && state.history.length > 0) {
    // Sample every 4th day across 2991 records
    const sample = state.history.filter((_, idx) => idx % 4 === 0);
    const labels = sample.map(d => d.date);
    const closeVals = sample.map(d => d.close);
    const meanVals = sample.map(d => d.rolling_mean);
    const stdVals = sample.map(d => d.rolling_std);

    if (typeof Chart !== 'undefined') {
      if (state.charts.rolling) state.charts.rolling.destroy();

      state.charts.rolling = new Chart(rollCanvas, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Closing Price ($)',
              data: closeVals,
              borderColor: '#38bdf8',
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.1
            },
            {
              label: '30-Day Rolling Mean ($)',
              data: meanVals,
              borderColor: '#f59e0b',
              borderWidth: 2,
              borderDash: [4, 4],
              pointRadius: 0
            },
            {
              label: '30-Day Rolling Std Dev (Volatility)',
              data: stdVals,
              borderColor: '#ec4899',
              borderWidth: 1.5,
              pointRadius: 0,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              labels: { color: THEME.textColor, font: { size: 11, family: 'Inter' } }
            }
          },
          scales: {
            x: {
              grid: { color: THEME.gridColor },
              ticks: { color: THEME.textColor, maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 9 } }
            },
            y: {
              grid: { color: THEME.gridColor },
              ticks: { color: THEME.textColor, callback: (v) => `$${v}` },
              title: { display: true, text: 'Price (USD)', color: THEME.textColor }
            },
            y1: {
              position: 'right',
              grid: { drawOnChartArea: false },
              ticks: { color: '#ec4899' },
              title: { display: true, text: 'Rolling Volatility ($)', color: '#ec4899' }
            }
          }
        }
      });
    }
  }

  // ACF & PACF Bar Charts
  if (state.acfPacf && typeof Chart !== 'undefined') {
    const acfCanvas = document.getElementById('acfCanvas');
    const pacfCanvas = document.getElementById('pacfCanvas');

    const lags = state.acfPacf.lags;
    const acfData = state.acfPacf.acf;
    const pacfData = state.acfPacf.pacf;
    const bound = state.acfPacf.confidence_bound;

    if (acfCanvas) {
      if (state.charts.acf) state.charts.acf.destroy();
      state.charts.acf = createAcfChart(acfCanvas, 'Autocorrelation (ACF)', lags, acfData, bound, '#38bdf8');
    }

    if (pacfCanvas) {
      if (state.charts.pacf) state.charts.pacf.destroy();
      state.charts.pacf = createAcfChart(pacfCanvas, 'Partial Autocorrelation (PACF)', lags, pacfData, bound, '#10b981');
    }
  }
}

function createAcfChart(canvas, title, lags, values, bound, barColor) {
  return new Chart(canvas, {
    type: 'bar',
    data: {
      labels: lags.map(l => `Lag ${l}`),
      datasets: [
        {
          type: 'bar',
          label: 'Correlation',
          data: values,
          backgroundColor: barColor,
          borderRadius: 2
        },
        {
          type: 'line',
          label: '+95% Conf Bound',
          data: lags.map(() => bound),
          borderColor: 'rgba(239, 68, 68, 0.7)',
          borderDash: [3, 3],
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          type: 'line',
          label: '-95% Conf Bound',
          data: lags.map(() => -bound),
          borderColor: 'rgba(239, 68, 68, 0.7)',
          borderDash: [3, 3],
          borderWidth: 1.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: THEME.gridColor },
          ticks: { color: THEME.textColor, maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 9 } }
        },
        y: {
          min: -0.4,
          max: 1.0,
          grid: { color: THEME.gridColor },
          ticks: { color: THEME.textColor, font: { family: 'JetBrains Mono', size: 9 } }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// TAB 3: RESIDUALS & ERROR DYNAMICS
// -------------------------------------------------------------
function renderResidualsTab() {
  if (!state.forecasts || state.forecasts.length === 0 || typeof Chart === 'undefined') return;

  const labels = state.forecasts.map(d => d.date);
  const arimaRes = state.forecasts.map(d => d.arima_residual);
  const lstmRes = state.forecasts.map(d => d.lstm_residual);

  // 1. Daily Residual Plot
  const resCanvas = document.getElementById('residualCanvas');
  if (resCanvas) {
    if (state.charts.residual) state.charts.residual.destroy();

    state.charts.residual = new Chart(resCanvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'ARIMA Residual (y - y_hat)',
            data: arimaRes,
            borderColor: '#f59e0b',
            borderWidth: 1.5,
            pointRadius: 0
          },
          {
            label: 'LSTM Residual (y - y_hat)',
            data: lstmRes,
            borderColor: '#10b981',
            borderWidth: 1.5,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.textColor } }
        },
        scales: {
          x: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor, maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor, callback: v => `$${v}` },
            title: { display: true, text: 'Residual Error ($)', color: THEME.textColor }
          }
        }
      }
    });
  }

  // 2. Cumulative Absolute Error
  const cumCanvas = document.getElementById('cumErrorCanvas');
  if (cumCanvas) {
    let cumArima = 0;
    let cumLstm = 0;
    const cumArimaArr = [];
    const cumLstmArr = [];

    state.forecasts.forEach(d => {
      cumArima += d.arima_abs_err;
      cumLstm += d.lstm_abs_err;
      cumArimaArr.push(cumArima);
      cumLstmArr.push(cumLstm);
    });

    if (state.charts.cumError) state.charts.cumError.destroy();

    state.charts.cumError = new Chart(cumCanvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Cumulative ARIMA Error ($)',
            data: cumArimaArr,
            borderColor: '#f59e0b',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            backgroundColor: 'rgba(245, 158, 11, 0.08)'
          },
          {
            label: 'Cumulative LSTM Error ($)',
            data: cumLstmArr,
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            backgroundColor: 'rgba(16, 185, 129, 0.08)'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.textColor } }
        },
        scales: {
          x: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor, maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor, callback: v => `$${v.toLocaleString()}` }
          }
        }
      }
    });
  }

  // 3. Error Distribution Histogram (Bins of Absolute Error)
  const histCanvas = document.getElementById('histCanvas');
  if (histCanvas) {
    const bins = ['0-5', '5-10', '10-20', '20-30', '30-50', '50-80', '80+'];
    const countArima = [0, 0, 0, 0, 0, 0, 0];
    const countLstm = [0, 0, 0, 0, 0, 0, 0];

    function getBinIdx(err) {
      if (err < 5) return 0;
      if (err < 10) return 1;
      if (err < 20) return 2;
      if (err < 30) return 3;
      if (err < 50) return 4;
      if (err < 80) return 5;
      return 6;
    }

    state.forecasts.forEach(d => {
      countArima[getBinIdx(d.arima_abs_err)]++;
      countLstm[getBinIdx(d.lstm_abs_err)]++;
    });

    if (state.charts.hist) state.charts.hist.destroy();

    state.charts.hist = new Chart(histCanvas, {
      type: 'bar',
      data: {
        labels: bins.map(b => `$${b} Error`),
        datasets: [
          {
            label: 'LSTM Days in Error Bin',
            data: countLstm,
            backgroundColor: '#10b981',
            borderRadius: 3
          },
          {
            label: 'ARIMA Days in Error Bin',
            data: countArima,
            backgroundColor: '#f59e0b',
            borderRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: THEME.textColor } }
        },
        scales: {
          x: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor, font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            grid: { color: THEME.gridColor },
            ticks: { color: THEME.textColor },
            title: { display: true, text: 'Number of Trading Days', color: THEME.textColor }
          }
        }
      }
    });
  }
}

// -------------------------------------------------------------
// TAB 5: VIVA & PROFESSOR DEFENSE ACCORDION
// -------------------------------------------------------------
function renderVivaAccordion() {
  const container = document.getElementById('qa-accordion-container');
  if (!container || !state.qaList || state.qaList.length === 0) return;

  container.innerHTML = '';

  state.qaList.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = `qa-card ${index === 0 ? 'open' : ''}`;

    card.innerHTML = `
      <div class="qa-header">
        <div class="qa-question-group">
          <span class="qa-badge">Q${item.id}</span>
          <span class="badge badge-blue" style="font-size: 0.65rem;">${item.category}</span>
          <span class="qa-title">${item.question}</span>
        </div>
        <span class="qa-icon">▼</span>
      </div>
      <div class="qa-body">
        <p>${item.answer}</p>
        <div class="qa-takeaway">
          <strong>Key Defense Takeaway:</strong> ${item.key_takeaway}
        </div>
      </div>
    `;

    card.querySelector('.qa-header').addEventListener('click', () => {
      card.classList.toggle('open');
    });

    container.appendChild(card);
  });
}

// -------------------------------------------------------------
// TAB 6: DATASET EXPLORER TABLE & PAGINATION
// -------------------------------------------------------------
function renderTablePage() {
  const tbody = document.getElementById('historical-table-body');
  const pageInfo = document.getElementById('table-page-info');
  if (!tbody || !state.filteredHistory) return;

  const total = state.filteredHistory.length;
  const totalPages = Math.max(1, Math.ceil(total / state.tablePageSize));
  if (state.tablePage > totalPages) state.tablePage = totalPages;

  const start = (state.tablePage - 1) * state.tablePageSize;
  const pageItems = state.filteredHistory.slice(start, start + state.tablePageSize);

  if (pageInfo) {
    pageInfo.innerText = `Showing records ${start + 1} to ${Math.min(start + state.tablePageSize, total)} of ${total.toLocaleString()} (Page ${state.tablePage} of ${totalPages})`;
  }

  if (pageItems.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">No matching trading records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = pageItems.map(row => `
    <tr>
      <td style="color: var(--accent-blue); font-weight: 600;">${row.date}</td>
      <td>$${row.open.toFixed(2)}</td>
      <td>$${row.high.toFixed(2)}</td>
      <td>$${row.low.toFixed(2)}</td>
      <td style="font-weight: 700;">$${row.close.toFixed(2)}</td>
      <td>${row.volume.toLocaleString()}</td>
      <td style="color: var(--text-secondary);">${row.rolling_mean ? '$' + row.rolling_mean.toFixed(2) : '-'}</td>
      <td style="color: ${row.diff >= 0 ? 'var(--accent-green)' : 'var(--accent-rose)'};">
        ${row.diff !== null ? (row.diff >= 0 ? '+' : '') + row.diff.toFixed(2) : '-'}
      </td>
    </tr>
  `).join('');
}

function exportDatasetCSV() {
  if (!state.history || state.history.length === 0) return;
  const headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', '30Day_Rolling_Mean', 'Daily_Diff'];
  const rows = state.history.map(r => [
    r.date, r.open, r.high, r.low, r.close, r.volume, r.rolling_mean || '', r.diff || ''
  ]);

  let csvContent = headers.join(',') + '\n';
  rows.forEach(row => {
    csvContent += row.join(',') + '\n';
  });

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `AAPL_historical_dataset_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// -------------------------------------------------------------
// NATIVE HTML5 CANVAS FALLBACK (WHEN OFFLINE WITHOUT CDN)
// -------------------------------------------------------------
function renderNativeCanvasLineChart(canvas, labels, datasets) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 40;

  ctx.clearRect(0, 0, w, h);

  // Compute min and max
  let minY = Infinity;
  let maxY = -Infinity;
  datasets.forEach(ds => {
    ds.data.forEach(v => {
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
    });
  });
  minY = Math.floor(minY * 0.95);
  maxY = Math.ceil(maxY * 1.05);

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  // Grid lines
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  const ySteps = 5;
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'right';

  for (let i = 0; i <= ySteps; i++) {
    const val = minY + (maxY - minY) * (i / ySteps);
    const y = padTop + plotH - (i / ySteps) * plotH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(`$${Math.round(val)}`, padLeft - 8, y + 3);
  }

  // Draw lines
  datasets.forEach(ds => {
    ctx.strokeStyle = ds.borderColor;
    ctx.lineWidth = ds.borderWidth || 2;
    if (ds.borderDash) {
      ctx.setLineDash(ds.borderDash);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ds.data.forEach((val, idx) => {
      const x = padLeft + (idx / (ds.data.length - 1)) * plotW;
      const y = padTop + plotH - ((val - minY) / (maxY - minY)) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
}
