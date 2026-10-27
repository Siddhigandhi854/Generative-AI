import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Upload, 
  Download, 
  Table as TableIcon, 
  BarChart2, 
  PieChart,
  Activity,
  Sparkles, 
  CheckCircle, 
  RefreshCw,
  FileText,
  Sliders,
  Filter,
  Layers,
  AlertTriangle,
  Flame
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SAMPLE_CSV = `prompt_id,domain,token_count,quality_score,perplexity,latency_ms,is_augmented
1,Vision,14,0.92,12.4,142,True
2,NLP,28,0.85,18.2,198,False
3,Audio,10,0.88,15.1,110,True
4,Vision,35,,22.7,285,False
5,Code,18,0.94,8.9,165,False
6,Video,22,0.91,14.6,210,True
7,NLP,28,0.85,18.2,198,False
8,Research,42,0.79,31.4,340,True
9,Vision,16,0.89,13.5,155,False
10,Audio,8,,9.8,95,True
11,Vision,19,0.95,11.2,175,False
12,Code,25,0.93,10.1,188,True
13,Video,31,0.87,19.8,260,False
14,NLP,15,0.90,14.0,130,True
15,Research,50,0.72,45.2,420,False
16,Code,12,0.96,7.5,125,True
17,Vision,20,0.91,12.9,180,False
18,Audio,11,0.86,16.4,115,True
19,Video,27,0.89,17.1,230,False
20,Research,38,0.81,25.6,310,True`;

export function Exp1DatasetAnalysis({ onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [operating, setOperating] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Chart view state
  const [activeChartTab, setActiveChartTab] = useState('distribution'); // 'distribution' | 'correlation' | 'outliers' | 'missing'
  const [selectedCol, setSelectedCol] = useState(null);
  const [chartType, setChartType] = useState('bar'); // 'bar' | 'line' | 'doughnut'

  // EDA Operation states
  const [activeEdaTab, setActiveEdaTab] = useState('clean'); // 'clean' | 'outliers' | 'scale' | 'encode'
  
  // Clean options
  const [missingStrategy, setMissingStrategy] = useState('impute_median');
  const [removeDuplicates, setRemoveDuplicates] = useState(true);

  // Outlier options
  const [outlierCol, setOutlierCol] = useState('all');
  const [outlierStrategy, setOutlierStrategy] = useState('clip');

  // Scaling options
  const [scaleMethod, setScaleMethod] = useState('minmax');
  const [scaleCol, setScaleCol] = useState('all');

  // Encoding options
  const [encodeCol, setEncodeCol] = useState('');
  const [encodeMethod, setEncodeMethod] = useState('one_hot');

  const handleFileUpload = async (file) => {
    setLoading(true);
    setError(null);
    setSuccessMsg(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/exp1/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Upload failed');
      }
      const result = await res.json();
      setData(result);
      if (result.column_names && result.column_names.length > 0) {
        setSelectedCol(result.column_names[0]);
      }
      if (result.categorical_columns && result.categorical_columns.length > 0) {
        setEncodeCol(result.categorical_columns[0]);
      }
      setSuccessMsg(`Dataset loaded: ${result.rows} rows, ${result.columns_count} features.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = () => {
    const blob = new Blob([SAMPLE_CSV], { type: 'text/csv' });
    const file = new File([blob], 'ai_prompt_dataset_v1.csv', { type: 'text/csv' });
    handleFileUpload(file);
  };

  const handleClean = async () => {
    if (!data?.dataset_id) return;
    setOperating(true);
    setError(null);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('dataset_id', data.dataset_id);
    formData.append('missing_strategy', missingStrategy);
    formData.append('remove_duplicates', String(removeDuplicates));

    try {
      const res = await fetch('/api/exp1/clean', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Cleaning operation failed');
      }
      const updated = await res.json();
      setData(updated);
      setSuccessMsg(`Cleaning complete: ${updated.rows_removed} duplicate/null rows processed.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setOperating(false);
    }
  };

  const handleOutlierOperation = async () => {
    if (!data?.dataset_id) return;
    setOperating(true);
    setError(null);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('dataset_id', data.dataset_id);
    formData.append('column', outlierCol);
    formData.append('strategy', outlierStrategy);

    try {
      const res = await fetch('/api/exp1/outliers/handle', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Outlier handling failed');
      }
      const updated = await res.json();
      setData(updated);
      setSuccessMsg(`Outlier handling (${outlierStrategy}) applied successfully.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setOperating(false);
    }
  };

  const handleScaleOperation = async () => {
    if (!data?.dataset_id) return;
    setOperating(true);
    setError(null);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('dataset_id', data.dataset_id);
    formData.append('method', scaleMethod);
    formData.append('column', scaleCol);

    try {
      const res = await fetch('/api/exp1/scale', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Feature scaling failed');
      }
      const updated = await res.json();
      setData(updated);
      setSuccessMsg(`Feature scaling (${scaleMethod === 'minmax' ? 'Min-Max [0,1]' : 'Standard Z-score'}) applied.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setOperating(false);
    }
  };

  const handleEncodeOperation = async () => {
    if (!data?.dataset_id || !encodeCol) return;
    setOperating(true);
    setError(null);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('dataset_id', data.dataset_id);
    formData.append('column', encodeCol);
    formData.append('method', encodeMethod);

    try {
      const res = await fetch('/api/exp1/encode', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Categorical encoding failed');
      }
      const updated = await res.json();
      setData(updated);
      setSuccessMsg(`Categorical encoding (${encodeMethod}) applied to '${encodeCol}'.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setOperating(false);
    }
  };

  // Prepare Chart.js data objects
  const getActiveDistributionChart = () => {
    if (!data || !selectedCol || !data.distributions || !data.distributions[selectedCol]) {
      return null;
    }
    const dist = data.distributions[selectedCol];
    const isNum = dist.type === 'numeric';

    const chartData = {
      labels: dist.labels,
      datasets: [
        {
          label: `${selectedCol} Frequency`,
          data: dist.values,
          backgroundColor: isNum
            ? 'rgba(24, 24, 27, 0.85)'
            : [
                '#18181b', '#3f3f46', '#71717a', '#a1a1aa',
                '#d4d4d8', '#52525b', '#27272a', '#09090b'
              ],
          borderColor: '#18181b',
          borderWidth: 1,
          borderRadius: 4,
          fill: true,
          tension: 0.35,
        }
      ]
    };

    return chartData;
  };

  const getMissingChartData = () => {
    if (!data || !data.missing_chart_data) return null;
    return {
      labels: data.missing_chart_data.labels,
      datasets: [
        {
          label: 'Missing Cells Count',
          data: data.missing_chart_data.counts,
          backgroundColor: data.missing_chart_data.counts.map(c => c > 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)'),
          borderColor: '#18181b',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    };
  };

  const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#18181b',
        titleFont: { size: 12 },
        bodyFont: { size: 11 },
        padding: 10,
        cornerRadius: 6
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { size: 10 }, color: '#71717a' }
      },
      y: {
        grid: { color: '#f4f4f5' },
        ticks: { font: { size: 10 }, color: '#71717a' },
        beginAtZero: true
      }
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Navigation & Header */}
      <div>
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-600 hover:text-zinc-950 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Lab Overview
        </button>

        <div className="border-b border-zinc-200 pb-5">
          <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider font-semibold">
            Experiment 1
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Comprehensive Dataset Analysis and Preprocessing for Generative AI using Python Libraries
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Interactive exploratory data analysis (EDA): statistical profiling, distribution curves, correlation heatmaps, IQR outlier handling, and feature scaling.
          </p>
        </div>
      </div>

      {/* Upload Dropzone & Quick Sample Loader */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 border-2 border-dashed border-zinc-200 hover:border-zinc-400 rounded-xl p-8 flex flex-col items-center justify-center bg-zinc-50/50 text-center transition-colors">
          <Upload className="w-8 h-8 text-zinc-400 mb-3" />
          <h3 className="text-sm font-semibold text-zinc-900">Upload CSV Dataset for EDA</h3>
          <p className="text-xs text-zinc-500 mt-1 max-w-sm">
            Drag and drop your training CSV file here, or click to browse.
          </p>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            className="mt-4 block text-xs text-zinc-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-zinc-950 file:text-white hover:file:bg-zinc-800 cursor-pointer"
          />
        </div>

        <div className="border border-zinc-200 rounded-xl p-6 bg-white flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-zinc-900 font-semibold text-sm">
              <FileText className="w-4 h-4 text-zinc-700" />
              Quick Sample AI Dataset
            </div>
            <p className="text-xs text-zinc-500 mt-2 leading-relaxed">
              Load an AI multi-modal prompt engineering benchmark containing numeric tokens, perplexity scores, latency stats, duplicates, and missing values.
            </p>
          </div>
          <button
            onClick={loadSampleData}
            disabled={loading}
            className="w-full mt-4 py-2 px-3 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-900 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            Load Sample AI Dataset
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
          <span>{error}</span>
        </div>
      )}

      {successMsg && (
        <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2">
          <CheckCircle className="w-4 h-4 shrink-0 text-emerald-600" />
          <span>{successMsg}</span>
        </div>
      )}

      {data && (
        <div className="space-y-8 animate-in fade-in duration-200">
          
          {/* Key Metric Overview Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-zinc-200 bg-white">
              <span className="text-xs text-zinc-500">Total Rows</span>
              <div className="text-2xl font-bold text-zinc-950 font-mono mt-1">{data.rows}</div>
            </div>
            <div className="p-4 rounded-xl border border-zinc-200 bg-white">
              <span className="text-xs text-zinc-500">Total Features</span>
              <div className="text-2xl font-bold text-zinc-950 font-mono mt-1">{data.columns_count}</div>
            </div>
            <div className="p-4 rounded-xl border border-zinc-200 bg-white">
              <span className="text-xs text-zinc-500">Missing Values</span>
              <div className={`text-2xl font-bold font-mono mt-1 ${data.total_missing > 0 ? 'text-amber-600' : 'text-emerald-600'}`}>
                {data.total_missing}
              </div>
            </div>
            <div className="p-4 rounded-xl border border-zinc-200 bg-white">
              <span className="text-xs text-zinc-500">Duplicate Rows</span>
              <div className={`text-2xl font-bold font-mono mt-1 ${data.duplicate_rows > 0 ? 'text-amber-600' : 'text-emerald-600'}`}>
                {data.duplicate_rows}
              </div>
            </div>
          </div>

          {/* ========================================== */}
          {/* VISUALIZATION SECTION: CHARTS & HEATMAPS   */}
          {/* ========================================== */}
          <div className="border border-zinc-200 rounded-2xl bg-white overflow-hidden shadow-sm">
            <div className="p-5 border-b border-zinc-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-zinc-50/50">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-zinc-950" />
                <div>
                  <h3 className="text-sm font-bold text-zinc-950">Exploratory Visualizations & Charts</h3>
                  <p className="text-xs text-zinc-500">Distributions, correlation heatmaps, missing value audits, and quantile spreads.</p>
                </div>
              </div>

              {/* Chart Navigation Tabs */}
              <div className="flex flex-wrap items-center gap-1.5 bg-zinc-100 p-1 rounded-lg border border-zinc-200">
                <button
                  type="button"
                  onClick={() => setActiveChartTab('distribution')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                    activeChartTab === 'distribution'
                      ? 'bg-white text-zinc-950 shadow-xs'
                      : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  Distributions
                </button>
                <button
                  type="button"
                  onClick={() => setActiveChartTab('correlation')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                    activeChartTab === 'correlation'
                      ? 'bg-white text-zinc-950 shadow-xs'
                      : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  <Flame className="w-3.5 h-3.5" />
                  Correlation Matrix
                </button>
                <button
                  type="button"
                  onClick={() => setActiveChartTab('outliers')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                    activeChartTab === 'outliers'
                      ? 'bg-white text-zinc-950 shadow-xs'
                      : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" />
                  Quantiles & Outliers
                </button>
                <button
                  type="button"
                  onClick={() => setActiveChartTab('missing')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                    activeChartTab === 'missing'
                      ? 'bg-white text-zinc-950 shadow-xs'
                      : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Missing Values
                </button>
              </div>
            </div>

            <div className="p-6">
              {/* 1. DISTRIBUTION CHARTS */}
              {activeChartTab === 'distribution' && (
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-zinc-700">Select Feature:</span>
                      <select
                        value={selectedCol || ''}
                        onChange={(e) => setSelectedCol(e.target.value)}
                        className="text-xs border border-zinc-300 rounded-lg px-3 py-1.5 bg-white text-zinc-900 font-medium focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        {data.column_names?.map((col) => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200">
                      <button
                        type="button"
                        onClick={() => setChartType('bar')}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                          chartType === 'bar' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                        }`}
                      >
                        Bar Histogram
                      </button>
                      <button
                        type="button"
                        onClick={() => setChartType('line')}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                          chartType === 'line' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                        }`}
                      >
                        Density Line
                      </button>
                      {data.distributions?.[selectedCol]?.type === 'categorical' && (
                        <button
                          type="button"
                          onClick={() => setChartType('doughnut')}
                          className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                            chartType === 'doughnut' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-900'
                          }`}
                        >
                          Doughnut
                        </button>
                      )}
                    </div>
                  </div>

                  {getActiveDistributionChart() ? (
                    <div className="h-72 w-full pt-2">
                      {chartType === 'bar' && (
                        <Bar data={getActiveDistributionChart()} options={commonChartOptions} />
                      )}
                      {chartType === 'line' && (
                        <Line data={getActiveDistributionChart()} options={commonChartOptions} />
                      )}
                      {chartType === 'doughnut' && (
                        <div className="h-full flex items-center justify-center">
                          <div className="w-64 h-64">
                            <Doughnut 
                              data={getActiveDistributionChart()} 
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } }
                                }
                              }} 
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-zinc-400">
                      No distribution data available for this column.
                    </div>
                  )}
                </div>
              )}

              {/* 2. CORRELATION HEATMAP MATRIX */}
              {activeChartTab === 'correlation' && (
                <div className="space-y-4">
                  {data.correlations?.columns?.length > 1 ? (
                    <div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-center text-xs">
                          <thead>
                            <tr>
                              <th className="p-3 text-left font-mono font-semibold text-zinc-700 bg-zinc-50 border border-zinc-200">
                                Feature
                              </th>
                              {data.correlations.columns.map((c) => (
                                <th key={c} className="p-3 font-mono font-semibold text-zinc-700 bg-zinc-50 border border-zinc-200">
                                  {c}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {data.correlations.columns.map((rowName, rIdx) => (
                              <tr key={rowName}>
                                <td className="p-3 text-left font-mono font-semibold text-zinc-900 bg-zinc-50 border border-zinc-200">
                                  {rowName}
                                </td>
                                {data.correlations.matrix[rIdx].map((val, cIdx) => {
                                  // Shading: 1.0 -> dark, 0.0 -> light, negative -> subtle blue
                                  let bgClass = 'bg-white text-zinc-900';
                                  if (val === 1.0) {
                                    bgClass = 'bg-zinc-950 text-white font-bold';
                                  } else if (val >= 0.7) {
                                    bgClass = 'bg-zinc-800 text-white font-semibold';
                                  } else if (val >= 0.4) {
                                    bgClass = 'bg-zinc-300 text-zinc-950 font-medium';
                                  } else if (val >= 0.1) {
                                    bgClass = 'bg-zinc-100 text-zinc-800';
                                  } else if (val <= -0.4) {
                                    bgClass = 'bg-sky-200 text-sky-950 font-medium';
                                  } else if (val < 0) {
                                    bgClass = 'bg-sky-50 text-sky-900';
                                  }

                                  return (
                                    <td
                                      key={cIdx}
                                      className={`p-3 font-mono border border-zinc-200 transition-colors ${bgClass}`}
                                      title={`Correlation(${rowName}, ${data.correlations.columns[cIdx]}): ${val}`}
                                    >
                                      {val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="text-[11px] text-zinc-400 mt-3 font-mono">
                        Pearson Correlation Matrix: values range from -1.0 (inverse relationship) to +1.0 (perfect linear correlation).
                      </p>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-zinc-400">
                      Need at least 2 numerical features to compute pairwise correlation.
                    </div>
                  )}
                </div>
              )}

              {/* 3. OUTLIER & QUANTILES SPREAD */}
              {activeChartTab === 'outliers' && (
                <div className="space-y-4">
                  {Object.keys(data.outliers || {}).length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-zinc-50 border border-zinc-200 text-zinc-600 font-semibold">
                          <tr>
                            <th className="py-2.5 px-4">Feature</th>
                            <th className="py-2.5 px-4">Min</th>
                            <th className="py-2.5 px-4">Q1 (25%)</th>
                            <th className="py-2.5 px-4">Median (50%)</th>
                            <th className="py-2.5 px-4">Q3 (75%)</th>
                            <th className="py-2.5 px-4">Max</th>
                            <th className="py-2.5 px-4">IQR Spread</th>
                            <th className="py-2.5 px-4">Outliers Count</th>
                            <th className="py-2.5 px-4">Skewness</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-200 font-mono border border-zinc-200">
                          {Object.entries(data.outliers).map(([col, o]) => (
                            <tr key={col} className="hover:bg-zinc-50/50">
                              <td className="py-2.5 px-4 font-semibold text-zinc-950 font-sans">{col}</td>
                              <td className="py-2.5 px-4 text-zinc-600">{o.min}</td>
                              <td className="py-2.5 px-4 text-zinc-700">{o.q1}</td>
                              <td className="py-2.5 px-4 font-bold text-zinc-950">{o.median}</td>
                              <td className="py-2.5 px-4 text-zinc-700">{o.q3}</td>
                              <td className="py-2.5 px-4 text-zinc-600">{o.max}</td>
                              <td className="py-2.5 px-4 text-zinc-700">{o.iqr}</td>
                              <td className="py-2.5 px-4">
                                <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                                  o.outlier_count > 0 ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                                }`}>
                                  {o.outlier_count} ({o.outlier_percentage}%)
                                </span>
                              </td>
                              <td className="py-2.5 px-4 text-zinc-600">{o.skewness}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-zinc-400">
                      No numerical features found to perform IQR outlier calculations.
                    </div>
                  )}
                </div>
              )}

              {/* 4. MISSING VALUES AUDIT */}
              {activeChartTab === 'missing' && (
                <div className="space-y-4">
                  {getMissingChartData() ? (
                    <div className="h-72 w-full pt-2">
                      <Bar data={getMissingChartData()} options={commonChartOptions} />
                    </div>
                  ) : (
                    <div className="py-12 text-center text-xs text-zinc-400">
                      No missing values found across all features.
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>

          {/* ========================================== */}
          {/* INTERACTIVE EDA OPERATIONS STUDIO          */}
          {/* ========================================== */}
          <div className="border border-zinc-200 rounded-2xl bg-white overflow-hidden shadow-sm">
            <div className="p-5 border-b border-zinc-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-zinc-50/50">
              <div className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-zinc-950" />
                <div>
                  <h3 className="text-sm font-bold text-zinc-950">Interactive EDA Operations Studio</h3>
                  <p className="text-xs text-zinc-500">Perform in-place transformations, outlier winsorization, scaling, and categorical encoding.</p>
                </div>
              </div>

              {/* Operations Tabs */}
              <div className="flex flex-wrap items-center gap-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200">
                <button
                  type="button"
                  onClick={() => setActiveEdaTab('clean')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    activeEdaTab === 'clean' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  1. Imputation & Duplicates
                </button>
                <button
                  type="button"
                  onClick={() => setActiveEdaTab('outliers')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    activeEdaTab === 'outliers' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  2. Outlier Treatment
                </button>
                <button
                  type="button"
                  onClick={() => setActiveEdaTab('scale')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    activeEdaTab === 'scale' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  3. Feature Scaling
                </button>
                <button
                  type="button"
                  onClick={() => setActiveEdaTab('encode')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    activeEdaTab === 'encode' ? 'bg-white text-zinc-950 shadow-xs' : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  4. Categorical Encoding
                </button>
              </div>
            </div>

            <div className="p-6">
              
              {/* TAB 1: CLEANING & IMPUTATION */}
              {activeEdaTab === 'clean' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Missing Value Imputation Strategy
                      </label>
                      <select
                        value={missingStrategy}
                        onChange={(e) => setMissingStrategy(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="impute_median">Median Imputation (Numeric) / Mode (Categorical)</option>
                        <option value="impute_mean">Mean Imputation (Numeric)</option>
                        <option value="impute_mode">Mode Imputation (Most Frequent)</option>
                        <option value="constant_zero">Constant Zero / "Unknown"</option>
                        <option value="drop">Drop Rows Containing Missing Values</option>
                      </select>
                    </div>

                    <div className="flex flex-col justify-end">
                      <label className="flex items-center gap-2 text-xs font-semibold text-zinc-800 cursor-pointer p-2.5 rounded-lg border border-zinc-200 bg-zinc-50">
                        <input
                          type="checkbox"
                          checked={removeDuplicates}
                          onChange={(e) => setRemoveDuplicates(e.target.checked)}
                          className="rounded border-zinc-300 text-zinc-950 focus:ring-0"
                        />
                        Deduplicate Rows (pd.drop_duplicates)
                      </label>
                    </div>

                    <div className="flex items-end">
                      <button
                        type="button"
                        onClick={handleClean}
                        disabled={operating}
                        className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {operating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                        Apply Imputation & Deduplication
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: OUTLIER TREATMENT */}
              {activeEdaTab === 'outliers' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Target Numerical Feature
                      </label>
                      <select
                        value={outlierCol}
                        onChange={(e) => setOutlierCol(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="all">All Numerical Features</option>
                        {data.numeric_columns?.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Outlier Handling Strategy (IQR)
                      </label>
                      <select
                        value={outlierStrategy}
                        onChange={(e) => setOutlierStrategy(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="clip">Winsorization (Clip to [Q1 - 1.5×IQR, Q3 + 1.5×IQR])</option>
                        <option value="drop">Trim / Drop Outlier Rows</option>
                      </select>
                    </div>

                    <div className="flex items-end">
                      <button
                        type="button"
                        onClick={handleOutlierOperation}
                        disabled={operating}
                        className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {operating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
                        Execute Outlier Treatment
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: FEATURE SCALING */}
              {activeEdaTab === 'scale' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Normalization / Scaling Method
                      </label>
                      <select
                        value={scaleMethod}
                        onChange={(e) => setScaleMethod(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="minmax">Min-Max Normalization: [0, 1]</option>
                        <option value="standard">StandardScaler (Z-Score: μ=0, σ=1)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Target Columns
                      </label>
                      <select
                        value={scaleCol}
                        onChange={(e) => setScaleCol(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="all">All Numeric Features</option>
                        {data.numeric_columns?.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-end">
                      <button
                        type="button"
                        onClick={handleScaleOperation}
                        disabled={operating}
                        className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {operating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sliders className="w-3.5 h-3.5" />}
                        Apply Feature Scaling
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: CATEGORICAL ENCODING */}
              {activeEdaTab === 'encode' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Select Categorical Feature
                      </label>
                      <select
                        value={encodeCol}
                        onChange={(e) => setEncodeCol(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        {data.categorical_columns?.length > 0 ? (
                          data.categorical_columns.map((c) => (
                            <option key={c} value={c}>{c}</option>
                          ))
                        ) : (
                          <option value="">No categorical columns found</option>
                        )}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
                        Encoding Technique
                      </label>
                      <select
                        value={encodeMethod}
                        onChange={(e) => setEncodeMethod(e.target.value)}
                        className="w-full text-xs border border-zinc-300 rounded-lg px-3 py-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                      >
                        <option value="one_hot">One-Hot Encoding (pd.get_dummies)</option>
                        <option value="label">Label Encoding (Ordinal Codes)</option>
                      </select>
                    </div>

                    <div className="flex items-end">
                      <button
                        type="button"
                        onClick={handleEncodeOperation}
                        disabled={operating || !encodeCol}
                        className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        {operating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Layers className="w-3.5 h-3.5" />}
                        Transform Categorical Feature
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Download Bar */}
              {data.download_url && (
                <div className="mt-5 pt-4 border-t border-zinc-100 flex items-center justify-between">
                  <span className="text-xs font-mono text-zinc-500">
                    Latest transformed dataset: {data.filename} ({data.rows} rows, {data.columns_count} columns)
                  </span>
                  <a
                    href={data.download_url}
                    download
                    className="px-4 py-2 text-xs font-semibold text-zinc-900 bg-zinc-100 hover:bg-zinc-200 border border-zinc-300 rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download Processed CSV
                  </a>
                </div>
              )}

            </div>
          </div>

          {/* ========================================== */}
          {/* STATISTICAL SUMMARY & SCHEMA AUDIT TABLES  */}
          {/* ========================================== */}
          <div className="border border-zinc-200 rounded-xl overflow-hidden bg-white shadow-sm">
            <div className="px-6 py-4 border-b border-zinc-200 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-950 flex items-center gap-2">
                <TableIcon className="w-4 h-4 text-zinc-700" />
                Column Schema & Missing Value Audit
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-500 font-medium">
                  <tr>
                    <th className="py-2.5 px-6">Column Name</th>
                    <th className="py-2.5 px-6">Data Type</th>
                    <th className="py-2.5 px-6">Unique Values</th>
                    <th className="py-2.5 px-6">Missing Values</th>
                    <th className="py-2.5 px-6">Missing %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100">
                  {data.columns_info?.map((col) => (
                    <tr key={col.name} className="hover:bg-zinc-50/50">
                      <td className="py-2.5 px-6 font-mono font-medium text-zinc-900">{col.name}</td>
                      <td className="py-2.5 px-6 font-mono text-zinc-600">{col.type}</td>
                      <td className="py-2.5 px-6 text-zinc-700">{col.unique_values}</td>
                      <td className="py-2.5 px-6 font-mono text-zinc-700">{col.missing_values}</td>
                      <td className="py-2.5 px-6">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-mono ${col.missing_percentage > 0 ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'}`}>
                          {col.missing_percentage}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Statistical Summary Table */}
          <div className="border border-zinc-200 rounded-xl overflow-hidden bg-white shadow-sm">
            <div className="px-6 py-4 border-b border-zinc-200">
              <h3 className="text-sm font-semibold text-zinc-950">Five-Number Statistical Summary</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-500 font-medium">
                  <tr>
                    <th className="py-2.5 px-4">Feature</th>
                    <th className="py-2.5 px-4">Count</th>
                    <th className="py-2.5 px-4">Mean</th>
                    <th className="py-2.5 px-4">Std</th>
                    <th className="py-2.5 px-4">Min</th>
                    <th className="py-2.5 px-4">25% (Q1)</th>
                    <th className="py-2.5 px-4">50% (Median)</th>
                    <th className="py-2.5 px-4">75% (Q3)</th>
                    <th className="py-2.5 px-4">Max</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 font-mono">
                  {data.stat_summary?.map((row) => (
                    <tr key={row.column} className="hover:bg-zinc-50/50">
                      <td className="py-2 px-4 font-semibold text-zinc-900">{row.column}</td>
                      <td className="py-2 px-4 text-zinc-600">{row.count}</td>
                      <td className="py-2 px-4 text-zinc-700">{row.mean !== "" ? row.mean : "-"}</td>
                      <td className="py-2 px-4 text-zinc-700">{row.std !== "" ? row.std : "-"}</td>
                      <td className="py-2 px-4 text-zinc-700">{row.min}</td>
                      <td className="py-2 px-4 text-zinc-700">{row['25%']}</td>
                      <td className="py-2 px-4 text-zinc-700">{row['50%']}</td>
                      <td className="py-2 px-4 text-zinc-700">{row['75%']}</td>
                      <td className="py-2 px-4 text-zinc-700">{row.max}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dataset Preview */}
          <div className="border border-zinc-200 rounded-xl overflow-hidden bg-white shadow-sm">
            <div className="px-6 py-4 border-b border-zinc-200 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-950">Active Dataset Preview (First 20 Rows)</h3>
              <span className="text-xs text-zinc-500 font-mono">{data.filename}</span>
            </div>
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs">
                <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-500 font-medium sticky top-0">
                  <tr>
                    {data.column_names?.map((col) => (
                      <th key={col} className="py-2.5 px-4 font-mono font-medium">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 font-mono">
                  {data.preview?.map((row, idx) => (
                    <tr key={idx} className="hover:bg-zinc-50/50">
                      {data.column_names?.map((col) => (
                        <td key={col} className="py-2 px-4 text-zinc-800 whitespace-nowrap">
                          {String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
