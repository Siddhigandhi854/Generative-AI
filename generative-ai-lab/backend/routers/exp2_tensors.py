import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Union
import numpy as np

_TF = None
_TF_ATTEMPTED = False

def get_tf():
    global _TF, _TF_ATTEMPTED
    if not _TF_ATTEMPTED:
        _TF_ATTEMPTED = True
        try:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            import tensorflow as tf
            _TF = tf
        except Exception:
            _TF = None
    return _TF

router = APIRouter(prefix="/api/exp2", tags=["Experiment 2: Tensor Operations"])

class TensorCreateRequest(BaseModel):
    creation_type: str  # 'constant', 'zeros', 'ones', 'random_normal', 'random_uniform'
    shape: List[int]
    values_raw: Optional[str] = None  # e.g. "[[1, 2], [3, 4]]"
    dtype: str = "float32"

class TensorOpRequest(BaseModel):
    op_type: str
    tensor_a: Any
    tensor_b: Optional[Any] = None
    axis: Optional[int] = None
    target_shape: Optional[List[int]] = None  # for reshape
    slice_start: Optional[List[int]] = None    # for slice
    slice_size: Optional[List[int]] = None     # for slice

def to_tf_dtype(dtype_str: str):
    tf = get_tf()
    if tf is None:
        return dtype_str
    mapping = {
        "float32": tf.float32,
        "float64": tf.float64,
        "int32": tf.int32,
        "int64": tf.int64,
        "bool": tf.bool
    }
    return mapping.get(dtype_str, tf.float32)

@router.get("/status")
def get_tensor_status():
    return {
        "engine": "TensorFlow 2.x (On-Demand) / NumPy",
        "tf_available": True,
        "version": "TensorFlow 2.x"
    }

@router.post("/create")
def create_tensor(req: TensorCreateRequest):
    try:
        tf = get_tf()
        dtype = to_tf_dtype(req.dtype)
        if req.creation_type == "constant":
            if not req.values_raw:
                raise ValueError("Values required for constant tensor.")
            parsed_vals = json.loads(req.values_raw)
            if tf is not None:
                t = tf.constant(parsed_vals, dtype=dtype)
            else:
                t = np.array(parsed_vals, dtype=req.dtype)
        elif req.creation_type == "zeros":
            if tf is not None:
                t = tf.zeros(req.shape, dtype=dtype)
            else:
                t = np.zeros(req.shape, dtype=req.dtype)
        elif req.creation_type == "ones":
            if tf is not None:
                t = tf.ones(req.shape, dtype=dtype)
            else:
                t = np.ones(req.shape, dtype=req.dtype)
        elif req.creation_type == "random_normal":
            if tf is not None:
                t = tf.random.normal(req.shape, mean=0.0, stddev=1.0, dtype=dtype)
            else:
                t = np.random.normal(0.0, 1.0, req.shape).astype(req.dtype)
        elif req.creation_type == "random_uniform":
            if tf is not None:
                t = tf.random.uniform(req.shape, minval=0.0, maxval=1.0, dtype=dtype)
            else:
                t = np.random.uniform(0.0, 1.0, req.shape).astype(req.dtype)
        else:
            raise ValueError(f"Unknown creation type: {req.creation_type}")

        if tf is not None:
            val_list = t.numpy().tolist()
            shape = list(t.shape)
            rank = len(shape)
            dtype_name = t.dtype.name
        else:
            val_list = t.tolist()
            shape = list(t.shape)
            rank = len(shape)
            dtype_name = str(t.dtype)

        return {
            "tensor": val_list,
            "shape": shape,
            "rank": rank,
            "dtype": dtype_name,
            "element_count": int(np.prod(shape)) if shape else 1
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/operate")
def operate_tensor(req: TensorOpRequest):
    try:
        tf = get_tf()
        a_arr = np.array(req.tensor_a, dtype=np.float32)
        b_arr = np.array(req.tensor_b, dtype=np.float32) if req.tensor_b is not None else None

        # Binary operations that strictly require Tensor B
        binary_ops = ["add", "subtract", "multiply", "divide", "power", "matmul", "concat", "stack"]
        if req.op_type in binary_ops and b_arr is None:
            raise ValueError(f"Operation '{req.op_type}' requires Tensor B.")

        if tf is not None:
            t_a = tf.constant(a_arr)
            t_b = tf.constant(b_arr) if b_arr is not None else None

            # --- LINEAR ALGEBRA ---
            if req.op_type == "matmul":
                res = tf.matmul(t_a, t_b)
            elif req.op_type == "transpose":
                res = tf.transpose(t_a)
            elif req.op_type == "det":
                res = tf.linalg.det(t_a)
            elif req.op_type == "inv":
                res = tf.linalg.inv(t_a)
            elif req.op_type == "trace":
                res = tf.linalg.trace(t_a)
            elif req.op_type == "norm":
                res = tf.norm(t_a)

            # --- ARITHMETIC ---
            elif req.op_type == "add":
                res = tf.add(t_a, t_b)
            elif req.op_type == "subtract":
                res = tf.subtract(t_a, t_b)
            elif req.op_type == "multiply":
                res = tf.multiply(t_a, t_b)
            elif req.op_type == "divide":
                res = tf.divide(t_a, t_b)
            elif req.op_type == "power":
                res = tf.pow(t_a, t_b)
            elif req.op_type == "abs":
                res = tf.abs(t_a)
            elif req.op_type == "negative":
                res = tf.negative(t_a)
            elif req.op_type == "square":
                res = tf.square(t_a)
            elif req.op_type == "sqrt":
                safe_a = tf.maximum(t_a, 0.0)
                res = tf.sqrt(safe_a)
            elif req.op_type == "exp":
                res = tf.exp(t_a)
            elif req.op_type == "log":
                safe_a = tf.maximum(t_a, 1e-7)
                res = tf.math.log(safe_a)

            # --- ACTIVATIONS ---
            elif req.op_type == "relu":
                res = tf.nn.relu(t_a)
            elif req.op_type == "sigmoid":
                res = tf.nn.sigmoid(t_a)
            elif req.op_type == "softmax":
                res = tf.nn.softmax(t_a, axis=req.axis if req.axis is not None else -1)
            elif req.op_type == "tanh":
                res = tf.nn.tanh(t_a)
            elif req.op_type == "gelu":
                res = tf.nn.gelu(t_a)
            elif req.op_type == "leaky_relu":
                res = tf.nn.leaky_relu(t_a, alpha=0.2)

            # --- REDUCTIONS ---
            elif req.op_type == "reduce_sum":
                res = tf.reduce_sum(t_a, axis=req.axis)
            elif req.op_type == "reduce_mean":
                res = tf.reduce_mean(t_a, axis=req.axis)
            elif req.op_type == "reduce_max":
                res = tf.reduce_max(t_a, axis=req.axis)
            elif req.op_type == "reduce_min":
                res = tf.reduce_min(t_a, axis=req.axis)
            elif req.op_type == "reduce_std":
                res = tf.math.reduce_std(t_a, axis=req.axis)
            elif req.op_type == "argmax":
                res = tf.math.argmax(t_a, axis=req.axis if req.axis is not None else -1)
            elif req.op_type == "argmin":
                res = tf.math.argmin(t_a, axis=req.axis if req.axis is not None else -1)

            # --- SHAPE MANIPULATION ---
            elif req.op_type == "reshape":
                if not req.target_shape:
                    raise ValueError("Target shape required for reshape.")
                res = tf.reshape(t_a, req.target_shape)
            elif req.op_type == "slice":
                if req.slice_start is None or req.slice_size is None:
                    raise ValueError("Slice start and size are required.")
                res = tf.slice(t_a, req.slice_start, req.slice_size)
            elif req.op_type == "squeeze":
                res = tf.squeeze(t_a, axis=[req.axis] if req.axis is not None else None)
            elif req.op_type == "expand_dims":
                res = tf.expand_dims(t_a, axis=req.axis if req.axis is not None else 0)
            elif req.op_type == "concat":
                res = tf.concat([t_a, t_b], axis=req.axis if req.axis is not None else 0)
            elif req.op_type == "stack":
                res = tf.stack([t_a, t_b], axis=req.axis if req.axis is not None else 0)
            else:
                raise ValueError(f"Unsupported operation: {req.op_type}")

            res_val = res.numpy().tolist()
            res_shape = list(res.shape)
            res_rank = len(res_shape)
            res_dtype = res.dtype.name

        else:
            # Complete NumPy vectorized fallback
            if req.op_type == "matmul":
                res = np.matmul(a_arr, b_arr)
            elif req.op_type == "transpose":
                res = np.transpose(a_arr)
            elif req.op_type == "det":
                res = np.linalg.det(a_arr)
            elif req.op_type == "inv":
                res = np.linalg.inv(a_arr)
            elif req.op_type == "trace":
                res = np.trace(a_arr)
            elif req.op_type == "norm":
                res = np.linalg.norm(a_arr)
            elif req.op_type == "add":
                res = np.add(a_arr, b_arr)
            elif req.op_type == "subtract":
                res = np.subtract(a_arr, b_arr)
            elif req.op_type == "multiply":
                res = np.multiply(a_arr, b_arr)
            elif req.op_type == "divide":
                res = np.divide(a_arr, b_arr)
            elif req.op_type == "power":
                res = np.power(a_arr, b_arr)
            elif req.op_type == "abs":
                res = np.abs(a_arr)
            elif req.op_type == "negative":
                res = np.negative(a_arr)
            elif req.op_type == "square":
                res = np.square(a_arr)
            elif req.op_type == "sqrt":
                res = np.sqrt(np.maximum(a_arr, 0.0))
            elif req.op_type == "exp":
                res = np.exp(a_arr)
            elif req.op_type == "log":
                res = np.log(np.maximum(a_arr, 1e-7))
            elif req.op_type == "relu":
                res = np.maximum(0.0, a_arr)
            elif req.op_type == "sigmoid":
                res = 1.0 / (1.0 + np.exp(-np.clip(a_arr, -30.0, 30.0)))
            elif req.op_type == "softmax":
                exp_a = np.exp(a_arr - np.max(a_arr, axis=req.axis if req.axis is not None else -1, keepdims=True))
                res = exp_a / np.sum(exp_a, axis=req.axis if req.axis is not None else -1, keepdims=True)
            elif req.op_type == "tanh":
                res = np.tanh(a_arr)
            elif req.op_type == "gelu":
                res = 0.5 * a_arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a_arr + 0.044715 * np.power(a_arr, 3))))
            elif req.op_type == "leaky_relu":
                res = np.where(a_arr > 0, a_arr, a_arr * 0.2)
            elif req.op_type == "reduce_sum":
                res = np.sum(a_arr, axis=req.axis)
            elif req.op_type == "reduce_mean":
                res = np.mean(a_arr, axis=req.axis)
            elif req.op_type == "reduce_max":
                res = np.max(a_arr, axis=req.axis)
            elif req.op_type == "reduce_min":
                res = np.min(a_arr, axis=req.axis)
            elif req.op_type == "reduce_std":
                res = np.std(a_arr, axis=req.axis)
            elif req.op_type == "argmax":
                res = np.argmax(a_arr, axis=req.axis if req.axis is not None else -1)
            elif req.op_type == "argmin":
                res = np.argmin(a_arr, axis=req.axis if req.axis is not None else -1)
            elif req.op_type == "reshape":
                res = np.reshape(a_arr, req.target_shape)
            elif req.op_type == "slice":
                slices = tuple(slice(st, st + sz) for st, sz in zip(req.slice_start, req.slice_size))
                res = a_arr[slices]
            elif req.op_type == "squeeze":
                res = np.squeeze(a_arr, axis=req.axis)
            elif req.op_type == "expand_dims":
                res = np.expand_dims(a_arr, axis=req.axis if req.axis is not None else 0)
            elif req.op_type == "concat":
                res = np.concatenate([a_arr, b_arr], axis=req.axis if req.axis is not None else 0)
            elif req.op_type == "stack":
                res = np.stack([a_arr, b_arr], axis=req.axis if req.axis is not None else 0)
            else:
                raise ValueError(f"Unsupported operation: {req.op_type}")

            res_val = res.tolist()
            res_shape = list(res.shape)
            res_rank = len(res_shape)
            res_dtype = str(res.dtype)

        # Round floats in result for clean presentation
        def clean_floats(obj):
            if isinstance(obj, float):
                return round(obj, 4)
            elif isinstance(obj, list):
                return [clean_floats(x) for x in obj]
            return obj

        clean_val = clean_floats(res_val)

        return {
            "result": clean_val,
            "shape": res_shape,
            "rank": res_rank,
            "dtype": res_dtype,
            "element_count": int(np.prod(res_shape)) if res_shape else 1,
            "engine": "TensorFlow 2.x" if (tf is not None) else "NumPy Engine"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
