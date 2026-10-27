import React, { useState, useEffect } from 'react';
import { ArrowLeft, Play, Cpu, Plus, RefreshCw, CheckCircle2, Sliders, Layers, Code, Zap } from 'lucide-react';

const OP_CATEGORIES = [
  {
    id: 'linalg',
    name: 'Linear Algebra',
    ops: [
      { id: 'matmul', name: 'tf.matmul', type: 'binary', desc: 'Matrix multiplication of Tensor A and Tensor B' },
      { id: 'transpose', name: 'tf.transpose', type: 'unary', desc: 'Permutes the dimensions of Tensor A' },
      { id: 'det', name: 'tf.linalg.det', type: 'unary', desc: 'Computes determinant of square matrix A' },
      { id: 'inv', name: 'tf.linalg.inv', type: 'unary', desc: 'Computes inverse of square matrix A' },
      { id: 'trace', name: 'tf.linalg.trace', type: 'unary', desc: 'Sum of the diagonal elements of Tensor A' },
      { id: 'norm', name: 'tf.norm', type: 'unary', desc: 'Computes the Euclidean / Frobenius norm of Tensor A' }
    ]
  },
  {
    id: 'arithmetic',
    name: 'Arithmetic',
    ops: [
      { id: 'add', name: 'tf.add', type: 'binary', desc: 'Element-wise addition: A + B' },
      { id: 'subtract', name: 'tf.subtract', type: 'binary', desc: 'Element-wise subtraction: A - B' },
      { id: 'multiply', name: 'tf.multiply', type: 'binary', desc: 'Element-wise multiplication: A * B' },
      { id: 'divide', name: 'tf.divide', type: 'binary', desc: 'Element-wise division: A / B' },
      { id: 'power', name: 'tf.pow', type: 'binary', desc: 'Element-wise power: A ^ B' },
      { id: 'abs', name: 'tf.abs', type: 'unary', desc: 'Computes absolute values |A|' },
      { id: 'square', name: 'tf.square', type: 'unary', desc: 'Computes element-wise square: A^2' },
      { id: 'sqrt', name: 'tf.sqrt', type: 'unary', desc: 'Computes element-wise square root' },
      { id: 'exp', name: 'tf.exp', type: 'unary', desc: 'Computes exponential e^A' },
      { id: 'log', name: 'tf.math.log', type: 'unary', desc: 'Computes natural logarithm ln(A)' }
    ]
  },
  {
    id: 'activations',
    name: 'Activations (DL)',
    ops: [
      { id: 'relu', name: 'tf.nn.relu', type: 'unary', desc: 'Rectified Linear Unit: max(0, x)' },
      { id: 'sigmoid', name: 'tf.nn.sigmoid', type: 'unary', desc: 'Sigmoid activation: 1 / (1 + e^-x)' },
      { id: 'softmax', name: 'tf.nn.softmax', type: 'unary', desc: 'Normalized probability distribution' },
      { id: 'tanh', name: 'tf.nn.tanh', type: 'unary', desc: 'Hyperbolic tangent activation' },
      { id: 'gelu', name: 'tf.nn.gelu', type: 'unary', desc: 'Gaussian Error Linear Unit (Transformers)' },
      { id: 'leaky_relu', name: 'tf.nn.leaky_relu', type: 'unary', desc: 'Leaky ReLU with alpha=0.2 (GANs)' }
    ]
  },
  {
    id: 'reductions',
    name: 'Reductions & Stats',
    ops: [
      { id: 'reduce_sum', name: 'tf.reduce_sum', type: 'unary', desc: 'Sum of tensor elements along axis' },
      { id: 'reduce_mean', name: 'tf.reduce_mean', type: 'unary', desc: 'Mean of tensor elements along axis' },
      { id: 'reduce_max', name: 'tf.reduce_max', type: 'unary', desc: 'Maximum element along axis' },
      { id: 'reduce_min', name: 'tf.reduce_min', type: 'unary', desc: 'Minimum element along axis' },
      { id: 'reduce_std', name: 'tf.math.reduce_std', type: 'unary', desc: 'Standard deviation along axis' },
      { id: 'argmax', name: 'tf.math.argmax', type: 'unary', desc: 'Index of maximum value along axis' },
      { id: 'argmin', name: 'tf.math.argmin', type: 'unary', desc: 'Index of minimum value along axis' }
    ]
  },
  {
    id: 'shape',
    name: 'Shape & Slicing',
    ops: [
      { id: 'reshape', name: 'tf.reshape', type: 'parametric', desc: 'Reshape tensor to target dimensions' },
      { id: 'slice', name: 'tf.slice', type: 'parametric', desc: 'Extract multidimensional sub-slice' },
      { id: 'squeeze', name: 'tf.squeeze', type: 'unary', desc: 'Removes dimensions of size 1' },
      { id: 'expand_dims', name: 'tf.expand_dims', type: 'parametric', desc: 'Inserts dimension of 1 at axis' },
      { id: 'concat', name: 'tf.concat', type: 'binary', desc: 'Concatenates Tensor A and B along axis' },
      { id: 'stack', name: 'tf.stack', type: 'binary', desc: 'Stacks Tensor A and B along new axis' }
    ]
  }
];

export function Exp2TensorOperations({ onBack }) {
  const [engineStatus, setEngineStatus] = useState(null);
  
  // Tensor A state
  const [createTypeA, setCreateTypeA] = useState('constant');
  const [shapeA, setShapeA] = useState('2, 2');
  const [valuesRawA, setValuesRawA] = useState('[[1, 2], [3, 4]]');
  const [tensorA, setTensorA] = useState([[1, 2], [3, 4]]);
  const [metaA, setMetaA] = useState({ shape: [2, 2], rank: 2, dtype: 'float32' });

  // Tensor B state
  const [createTypeB, setCreateTypeB] = useState('constant');
  const [shapeB, setShapeB] = useState('2, 2');
  const [valuesRawB, setValuesRawB] = useState('[[5, 6], [7, 8]]');
  const [tensorB, setTensorB] = useState([[5, 6], [7, 8]]);
  const [metaB, setMetaB] = useState({ shape: [2, 2], rank: 2, dtype: 'float32' });

  // Active Category & Operation
  const [activeCategory, setActiveCategory] = useState('linalg');
  const [selectedOp, setSelectedOp] = useState('matmul');
  
  // Parameters for specific ops
  const [targetShape, setTargetShape] = useState('4');
  const [sliceStart, setSliceStart] = useState('0, 0');
  const [sliceSize, setSliceSize] = useState('1, 2');
  const [axisParam, setAxisParam] = useState('0');
  
  // Result state
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/exp2/status')
      .then((res) => res.json())
      .then((d) => setEngineStatus(d))
      .catch((e) => console.error(e));
  }, []);

  const currentOpMeta = OP_CATEGORIES.flatMap(c => c.ops).find(o => o.id === selectedOp) || {
    id: selectedOp,
    name: `tf.${selectedOp}`,
    type: 'binary',
    desc: ''
  };

  const isUnaryOp = currentOpMeta.type === 'unary';
  const isBinaryOp = currentOpMeta.type === 'binary';

  const handleCreateTensor = async (target) => {
    setError(null);
    const isA = target === 'A';
    const createType = isA ? createTypeA : createTypeB;
    const shapeStr = isA ? shapeA : shapeB;
    const valuesRaw = isA ? valuesRawA : valuesRawB;

    const parsedShape = shapeStr.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));

    try {
      const res = await fetch('/api/exp2/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          creation_type: createType,
          shape: parsedShape,
          values_raw: valuesRaw,
          dtype: 'float32'
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Tensor creation failed');
      }

      const data = await res.json();
      if (isA) {
        setTensorA(data.tensor);
        setMetaA({ shape: data.shape, rank: data.rank, dtype: data.dtype });
      } else {
        setTensorB(data.tensor);
        setMetaB({ shape: data.shape, rank: data.rank, dtype: data.dtype });
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRunOperation = async () => {
    if (!tensorA) {
      setError('Tensor A is required.');
      return;
    }
    setRunning(true);
    setError(null);

    const payload = {
      op_type: selectedOp,
      tensor_a: tensorA
    };

    if (isBinaryOp) {
      payload.tensor_b = tensorB;
    }

    if (selectedOp === 'reshape') {
      payload.target_shape = targetShape.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
    } else if (selectedOp === 'slice') {
      payload.slice_start = sliceStart.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      payload.slice_size = sliceSize.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
    } else if (['concat', 'stack', 'expand_dims', 'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min', 'reduce_std', 'argmax', 'argmin', 'softmax'].includes(selectedOp)) {
      if (axisParam.trim() !== '') {
        const parsedAxis = parseInt(axisParam.trim(), 10);
        if (!isNaN(parsedAxis)) {
          payload.axis = parsedAxis;
        }
      }
    }

    try {
      const res = await fetch('/api/exp2/operate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Tensor operation failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  const loadPreset = (preset) => {
    setError(null);
    if (preset === 'matmul') {
      setValuesRawA('[[1, 2], [3, 4]]');
      setTensorA([[1, 2], [3, 4]]);
      setShapeA('2, 2');
      setMetaA({ shape: [2, 2], rank: 2, dtype: 'float32' });

      setValuesRawB('[[2, 0], [1, 2]]');
      setTensorB([[2, 0], [1, 2]]);
      setShapeB('2, 2');
      setMetaB({ shape: [2, 2], rank: 2, dtype: 'float32' });

      setActiveCategory('linalg');
      setSelectedOp('matmul');
    } else if (preset === 'softmax') {
      setValuesRawA('[[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]]');
      setTensorA([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]]);
      setShapeA('2, 3');
      setMetaA({ shape: [2, 3], rank: 2, dtype: 'float32' });

      setActiveCategory('activations');
      setSelectedOp('softmax');
      setAxisParam('-1');
    } else if (preset === 'relu') {
      setValuesRawA('[[-2.5, 1.2, -0.4], [3.1, -4.0, 0.8]]');
      setTensorA([[-2.5, 1.2, -0.4], [3.1, -4.0, 0.8]]);
      setShapeA('2, 3');
      setMetaA({ shape: [2, 3], rank: 2, dtype: 'float32' });

      setActiveCategory('activations');
      setSelectedOp('relu');
    } else if (preset === 'reshape') {
      setValuesRawA('[[1, 2, 3], [4, 5, 6]]');
      setTensorA([[1, 2, 3], [4, 5, 6]]);
      setShapeA('2, 3');
      setMetaA({ shape: [2, 3], rank: 2, dtype: 'float32' });

      setActiveCategory('shape');
      setSelectedOp('reshape');
      setTargetShape('3, 2');
    } else if (preset === 'det') {
      setValuesRawA('[[4, 7], [2, 6]]');
      setTensorA([[4, 7], [2, 6]]);
      setShapeA('2, 2');
      setMetaA({ shape: [2, 2], rank: 2, dtype: 'float32' });

      setActiveCategory('linalg');
      setSelectedOp('det');
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
            Experiment 2
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Basic Python Operations and TensorFlow 2 Tensor Operations
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Interactive execution engine for tensor algebra, linear systems, neural activations, reductions, and multi-axis slicing using TensorFlow 2.
          </p>
        </div>
      </div>

      {/* Presets Bar */}
      <div className="flex flex-wrap items-center gap-2 p-3 bg-zinc-50 rounded-xl border border-zinc-200">
        <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider px-2 flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-zinc-700" />
          Quick Presets:
        </span>
        <button
          onClick={() => loadPreset('matmul')}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-zinc-200 hover:bg-zinc-100 text-zinc-800 transition-colors shadow-xs"
        >
          2×2 Matrix Multiplication (tf.matmul)
        </button>
        <button
          onClick={() => loadPreset('softmax')}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-zinc-200 hover:bg-zinc-100 text-zinc-800 transition-colors shadow-xs"
        >
          Softmax Probability Distribution (tf.nn.softmax)
        </button>
        <button
          onClick={() => loadPreset('relu')}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-zinc-200 hover:bg-zinc-100 text-zinc-800 transition-colors shadow-xs"
        >
          ReLU Activation (tf.nn.relu)
        </button>
        <button
          onClick={() => loadPreset('det')}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-zinc-200 hover:bg-zinc-100 text-zinc-800 transition-colors shadow-xs"
        >
          Matrix Determinant (tf.linalg.det)
        </button>
        <button
          onClick={() => loadPreset('reshape')}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-zinc-200 hover:bg-zinc-100 text-zinc-800 transition-colors shadow-xs"
        >
          Reshape [2, 3] → [3, 2] (tf.reshape)
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs font-mono">
          {error}
        </div>
      )}

      {/* Tensor Builders: Tensor A and Tensor B */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Tensor A Card */}
        <div className="border border-zinc-200 rounded-xl p-6 bg-white space-y-4 shadow-xs">
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
            <h3 className="text-sm font-semibold text-zinc-950 font-mono flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-zinc-950 text-white text-[11px] font-bold flex items-center justify-center font-mono">A</span>
              Tensor A
            </h3>
            <div className="flex items-center gap-2 text-xs font-mono text-zinc-500">
              <span className="px-2 py-0.5 rounded bg-zinc-100">Shape: [{metaA.shape.join(', ')}]</span>
              <span className="px-2 py-0.5 rounded bg-zinc-100">Rank: {metaA.rank}</span>
            </div>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Creation Initializer</label>
                <select
                  value={createTypeA}
                  onChange={(e) => setCreateTypeA(e.target.value)}
                  className="w-full text-xs border border-zinc-300 rounded-lg px-2.5 py-2 bg-white text-zinc-800 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                >
                  <option value="constant">Custom Values (Constant)</option>
                  <option value="zeros">Zeros (tf.zeros)</option>
                  <option value="ones">Ones (tf.ones)</option>
                  <option value="random_normal">Random Normal (μ=0, σ=1)</option>
                  <option value="random_uniform">Random Uniform [0, 1)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Dimensions (Shape)</label>
                <input
                  type="text"
                  value={shapeA}
                  onChange={(e) => setShapeA(e.target.value)}
                  placeholder="e.g. 2, 2"
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-2.5 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                />
              </div>
            </div>

            {createTypeA === 'constant' && (
              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Tensor Values (JSON array)</label>
                <input
                  type="text"
                  value={valuesRawA}
                  onChange={(e) => setValuesRawA(e.target.value)}
                  placeholder="[[1, 2], [3, 4]]"
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-2.5 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                />
              </div>
            )}

            <button
              onClick={() => handleCreateTensor('A')}
              className="w-full py-2 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-900 transition-colors"
            >
              Update Tensor A
            </button>

            {/* Tensor A Display */}
            <div className="mt-2 p-3 bg-zinc-50 rounded-lg border border-zinc-200 font-mono text-xs overflow-x-auto text-zinc-900 max-h-40">
              <pre>{JSON.stringify(tensorA, null, 2)}</pre>
            </div>
          </div>
        </div>

        {/* Tensor B Card */}
        <div className={`border border-zinc-200 rounded-xl p-6 bg-white space-y-4 shadow-xs transition-opacity ${
          isUnaryOp ? 'opacity-50' : 'opacity-100'
        }`}>
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
            <h3 className="text-sm font-semibold text-zinc-950 font-mono flex items-center gap-2">
              <span className="w-5 h-5 rounded-md bg-zinc-200 text-zinc-900 text-[11px] font-bold flex items-center justify-center font-mono">B</span>
              Tensor B {isUnaryOp && <span className="text-xs font-normal text-zinc-400 font-sans">(Not needed for unary op)</span>}
            </h3>
            <div className="flex items-center gap-2 text-xs font-mono text-zinc-500">
              <span className="px-2 py-0.5 rounded bg-zinc-100">Shape: [{metaB.shape.join(', ')}]</span>
              <span className="px-2 py-0.5 rounded bg-zinc-100">Rank: {metaB.rank}</span>
            </div>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Creation Initializer</label>
                <select
                  disabled={isUnaryOp}
                  value={createTypeB}
                  onChange={(e) => setCreateTypeB(e.target.value)}
                  className="w-full text-xs border border-zinc-300 rounded-lg px-2.5 py-2 bg-white text-zinc-800 focus:outline-none focus:ring-1 focus:ring-zinc-900 disabled:opacity-50"
                >
                  <option value="constant">Custom Values (Constant)</option>
                  <option value="zeros">Zeros (tf.zeros)</option>
                  <option value="ones">Ones (tf.ones)</option>
                  <option value="random_normal">Random Normal (μ=0, σ=1)</option>
                  <option value="random_uniform">Random Uniform [0, 1)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Dimensions (Shape)</label>
                <input
                  type="text"
                  disabled={isUnaryOp}
                  value={shapeB}
                  onChange={(e) => setShapeB(e.target.value)}
                  placeholder="e.g. 2, 2"
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-2.5 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-900 disabled:opacity-50"
                />
              </div>
            </div>

            {createTypeB === 'constant' && (
              <div>
                <label className="block text-xs font-semibold text-zinc-600 mb-1">Tensor Values (JSON array)</label>
                <input
                  type="text"
                  disabled={isUnaryOp}
                  value={valuesRawB}
                  onChange={(e) => setValuesRawB(e.target.value)}
                  placeholder="[[5, 6], [7, 8]]"
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-2.5 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-900 disabled:opacity-50"
                />
              </div>
            )}

            <button
              disabled={isUnaryOp}
              onClick={() => handleCreateTensor('B')}
              className="w-full py-2 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-900 transition-colors disabled:opacity-50"
            >
              Update Tensor B
            </button>

            {/* Tensor B Display */}
            <div className="mt-2 p-3 bg-zinc-50 rounded-lg border border-zinc-200 font-mono text-xs overflow-x-auto text-zinc-900 max-h-40">
              <pre>{JSON.stringify(tensorB, null, 2)}</pre>
            </div>
          </div>
        </div>

      </div>

      {/* Operation Selection & Execution Studio */}
      <div className="border border-zinc-200 rounded-2xl p-6 bg-white space-y-6 shadow-sm">
        
        {/* Category Tabs */}
        <div>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100">
            <div>
              <h3 className="text-sm font-semibold text-zinc-950 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-zinc-800" />
                TensorFlow 2 Operation Catalog
              </h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                Select an operation category and choose from the complete suite of mathematical transformations.
              </p>
            </div>

            {/* Category Switcher */}
            <div className="flex flex-wrap items-center gap-1.5 bg-zinc-100 p-1 rounded-xl border border-zinc-200">
              {OP_CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                    activeCategory === cat.id
                      ? 'bg-white text-zinc-950 shadow-xs'
                      : 'text-zinc-600 hover:text-zinc-950'
                  }`}
                >
                  {cat.name}
                </button>
              ))}
            </div>
          </div>

          {/* Operations Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 pt-4">
            {OP_CATEGORIES.find(c => c.id === activeCategory)?.ops.map((op) => (
              <button
                key={op.id}
                onClick={() => setSelectedOp(op.id)}
                className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  selectedOp === op.id
                    ? 'border-zinc-950 bg-zinc-950 text-white shadow-sm'
                    : 'border-zinc-200 bg-white hover:border-zinc-400 hover:bg-zinc-50 text-zinc-900'
                }`}
              >
                <div>
                  <span className={`text-[10px] font-mono uppercase tracking-wider block font-semibold ${
                    selectedOp === op.id ? 'text-zinc-400' : 'text-zinc-500'
                  }`}>
                    {op.type}
                  </span>
                  <span className="text-xs font-mono font-bold mt-1 block">
                    {op.name}
                  </span>
                </div>
                <span className={`text-[11px] line-clamp-2 mt-2 leading-tight ${
                  selectedOp === op.id ? 'text-zinc-300' : 'text-zinc-500'
                }`}>
                  {op.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Dynamic Parameter Settings */}
        {selectedOp === 'reshape' && (
          <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <label className="text-xs font-semibold text-zinc-800">Target Shape Vector:</label>
            <input
              type="text"
              value={targetShape}
              onChange={(e) => setTargetShape(e.target.value)}
              placeholder="e.g. 4 or 1, 4"
              className="text-xs font-mono border border-zinc-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900 bg-white"
            />
            <span className="text-xs text-zinc-500">
              (Total volume must match Tensor A count of {metaA.shape.reduce((a, b) => a * b, 1)} elements)
            </span>
          </div>
        )}

        {selectedOp === 'slice' && (
          <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">Slice Begin Coordinates [begin]:</label>
              <input
                type="text"
                value={sliceStart}
                onChange={(e) => setSliceStart(e.target.value)}
                placeholder="0, 0"
                className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900 bg-white"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-700 mb-1">Slice Extent Sizes [size]:</label>
              <input
                type="text"
                value={sliceSize}
                onChange={(e) => setSliceSize(e.target.value)}
                placeholder="1, 2"
                className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900 bg-white"
              />
            </div>
          </div>
        )}

        {['concat', 'stack', 'expand_dims', 'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min', 'reduce_std', 'argmax', 'argmin', 'softmax'].includes(selectedOp) && (
          <div className="p-4 bg-zinc-50 rounded-xl border border-zinc-200 flex items-center gap-4">
            <label className="text-xs font-semibold text-zinc-800">Reduction / Transformation Axis:</label>
            <input
              type="text"
              value={axisParam}
              onChange={(e) => setAxisParam(e.target.value)}
              placeholder="0 (or -1 for last axis)"
              className="w-28 text-xs font-mono border border-zinc-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900 bg-white"
            />
            <span className="text-xs text-zinc-500">
              Default: 0 for concatenation/stacking, -1 for softmax probability normalization.
            </span>
          </div>
        )}

        {/* Execution Button */}
        <button
          onClick={handleRunOperation}
          disabled={running}
          className="w-full py-3.5 rounded-xl bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50 shadow-sm"
        >
          {running ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
          Execute Operation: {currentOpMeta.name}
        </button>

        {/* Operation Execution Result */}
        {result && (
          <div className="p-6 rounded-xl border border-zinc-200 bg-zinc-50/50 space-y-4 animate-in fade-in duration-200">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-200 pb-3">
              <div className="flex items-center gap-2 text-zinc-950 font-semibold text-sm">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Tensor Operation Result</span>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-zinc-700">
                <span className="px-2 py-0.5 rounded bg-white border border-zinc-200">Shape: [{result.shape.join(', ')}]</span>
                <span className="px-2 py-0.5 rounded bg-white border border-zinc-200">Rank: {result.rank}</span>
                <span className="px-2 py-0.5 rounded bg-white border border-zinc-200">Dtype: {result.dtype}</span>
                <span className="px-2 py-0.5 rounded bg-white border border-zinc-200">Elements: {result.element_count}</span>
              </div>
            </div>

            <div className="p-4 bg-white rounded-xl border border-zinc-200 font-mono text-xs overflow-x-auto text-zinc-900 shadow-xs">
              <pre>{JSON.stringify(result.result, null, 2)}</pre>
            </div>
          </div>
        )}

      </div>

    </div>
  );
}
