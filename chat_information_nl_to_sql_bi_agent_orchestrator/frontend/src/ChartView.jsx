import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  ArcElement,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Pie, Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  ArcElement,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

/**
 * AutoChart:
 * - dataRows: array of {col1: v1, col2: v2, ...}
 * - tries to detect a categorical column (string) and one numeric column
 * - If exactly one categorical + one numeric column -> bar chart
 * - If single categorical with numeric counts -> pie chart
 * - If timeseries-like numeric sequence (index numeric) -> line chart
 */
export default function ChartView({ dataRows, chartTypeOverride = null }) {
  if (!dataRows || dataRows.length === 0) return null;

  // derive column names
  const cols = Object.keys(dataRows[0]);

  // helper to detect numeric column
  const isNumeric = (val) => {
    return val !== null && val !== undefined && !isNaN(Number(val));
  };

  // compute column stats
  const colStats = {};
  for (const c of cols) {
    let numericCount = 0;
    for (let i = 0; i < Math.min(dataRows.length, 50); i++) {
      if (isNumeric(dataRows[i][c])) numericCount++;
    }
    colStats[c] = { numericCount, sampleValue: dataRows[0][c] };
  }

  // find one categorical and one numeric candidate
  let catCol = null;
  let numCol = null;

  // prefer patterns: (cat, num)
  for (const c of cols) {
    if (colStats[c].numericCount >= Math.min(5, Math.floor(dataRows.length / 2))) {
      // numeric-ish
      if (!numCol) numCol = c;
    } else {
      if (!catCol) catCol = c;
    }
  }

  // fallback: if only two cols and one is numeric -> treat other as category
  if (!catCol && cols.length === 2 && numCol) {
    catCol = cols.find((c) => c !== numCol);
  }

  // Build data for chart
  if (chartTypeOverride) {
    // allow manual override (e.g., "bar", "pie", "line")
  }

  // If we have a category + number -> Bar or Pie
  if (catCol && numCol) {
    const labels = dataRows.map((r) => String(r[catCol] ?? ""));
    const values = dataRows.map((r) => Number(r[numCol] ?? 0));

    const chartData = {
      labels,
      datasets: [
        {
          label: numCol,
          data: values,
          backgroundColor: labels.map((_, i) => `hsl(${(i * 47) % 360} 70% 60%)`),
        },
      ],
    };

    // choose pie if relatively few categories (<10), else bar
    if (chartTypeOverride === "pie" || (labels.length <= 10 && chartTypeOverride !== "bar")) {
      return (
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          <h4>Pie chart: {numCol} by {catCol}</h4>
          <Pie data={chartData} />
        </div>
      );
    }

    return (
      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        <h4>Bar chart: {numCol} by {catCol}</h4>
        <Bar data={chartData} options={{ responsive: true }} />
      </div>
    );
  }

  // If single numeric column across rows -> show line
  if (!catCol && numCol && cols.length === 1) {
    const labels = dataRows.map((_, i) => i + 1);
    const values = dataRows.map((r) => Number(r[cols[0]] ?? 0));
    return (
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <h4>Line chart: {cols[0]}</h4>
        <Line data={{ labels, datasets: [{ label: cols[0], data: values }] }} />
      </div>
    );
  }

  // no obvious chart
  return null;
}
