import io
import os
import uuid
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List, Dict, Any
from config import config

router = APIRouter(prefix="/api/exp1", tags=["Experiment 1: Dataset Analysis"])

# In-memory storage for uploaded and cleaned datasets
DATASET_STORE: Dict[str, pd.DataFrame] = {}

def analyze_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_cnt = int(df[col].nunique())
        null_cnt = int(df[col].isnull().sum())
        columns_info.append({
            "name": col,
            "type": dtype,
            "is_numeric": col in numeric_cols,
            "unique_values": unique_cnt,
            "missing_values": null_cnt,
            "missing_percentage": round((null_cnt / max(1, rows)) * 100, 2)
        })
    
    # Statistical Summary
    desc = df.describe(include="all").to_dict()
    stat_summary = []
    for col in df.columns:
        col_stats = {
            "column": col,
            "count": str(desc.get(col, {}).get("count", "")),
            "mean": round(float(desc[col]["mean"]), 3) if col in desc and pd.notnull(desc[col].get("mean")) else "",
            "std": round(float(desc[col]["std"]), 3) if col in desc and pd.notnull(desc[col].get("std")) else "",
            "min": str(desc.get(col, {}).get("min", "")),
            "25%": str(desc.get(col, {}).get("25%", "")),
            "50%": str(desc.get(col, {}).get("50%", "")),
            "75%": str(desc.get(col, {}).get("75%", "")),
            "max": str(desc.get(col, {}).get("max", ""))
        }
        stat_summary.append(col_stats)
    
    # Distributions for columns
    distributions = {}
    for col in df.columns[:15]:
        if col in numeric_cols:
            clean_series = df[col].dropna()
            if len(clean_series) > 0:
                counts, bin_edges = np.histogram(clean_series, bins=min(10, max(4, len(clean_series) // 2)))
                distributions[col] = {
                    "type": "numeric",
                    "labels": [f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}" for i in range(len(counts))],
                    "values": [int(c) for c in counts]
                }
        else:
            top_cats = df[col].value_counts().head(10).to_dict()
            distributions[col] = {
                "type": "categorical",
                "labels": [str(k) for k in top_cats.keys()],
                "values": [int(v) for v in top_cats.values()]
            }
            
    # Correlation Matrix across numeric columns
    correlations = {"columns": [], "matrix": []}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr().fillna(0)
        correlations = {
            "columns": numeric_cols,
            "matrix": [[round(float(v), 3) for v in row] for row in corr_df.values]
        }
        
    # Outlier Analysis & IQR Metrics
    outliers = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) >= 4:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_cnt = int(outlier_mask.sum())
            outliers[col] = {
                "q1": round(q1, 3),
                "median": round(float(series.median()), 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
                "outlier_count": outlier_cnt,
                "outlier_percentage": round((outlier_cnt / max(1, len(series))) * 100, 2),
                "skewness": round(float(series.skew()), 3) if len(series) > 2 else 0.0,
                "kurtosis": round(float(series.kurt()), 3) if len(series) > 3 else 0.0,
            }
            
    # Missing Values Distribution Chart Data
    missing_chart_data = {
        "labels": list(df.columns),
        "counts": [int(df[c].isnull().sum()) for c in df.columns],
        "percentages": [round((int(df[c].isnull().sum()) / max(1, rows)) * 100, 2) for c in df.columns]
    }
            
    # Preview (first 20 rows)
    preview_df = df.head(20).astype(object).fillna("NaN")
    preview_rows = preview_df.to_dict(orient="records")
    
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": rows,
        "columns_count": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_names": list(df.columns),
        "columns_info": columns_info,
        "total_missing": total_missing,
        "missing_per_col": missing_per_col,
        "missing_chart_data": missing_chart_data,
        "duplicate_rows": duplicate_rows,
        "stat_summary": stat_summary,
        "distributions": distributions,
        "correlations": correlations,
        "outliers": outliers,
        "preview": preview_rows
    }

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    
    dataset_id = str(uuid.uuid4())
    DATASET_STORE[dataset_id] = df
    
    return analyze_dataframe(df, dataset_id, file.filename)

@router.post("/clean")
async def clean_dataset(
    dataset_id: str = Form(...),
    missing_strategy: str = Form("impute"),  # 'impute_mean', 'impute_median', 'impute_mode', 'constant_zero', 'drop'
    remove_duplicates: bool = Form(True)
):
    if dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")
    
    df = DATASET_STORE[dataset_id].copy()
    initial_rows = len(df)
    
    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates()
        
    # Handle missing values
    if missing_strategy == "drop":
        df = df.dropna()
    elif missing_strategy in ["impute", "impute_median"]:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
    elif missing_strategy == "impute_mean":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
    elif missing_strategy == "impute_mode":
        for col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
    elif missing_strategy == "constant_zero":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Unknown")
                    
    cleaned_id = str(uuid.uuid4())
    DATASET_STORE[cleaned_id] = df
    
    # Save cleaned file for download
    output_filename = f"cleaned_dataset_{cleaned_id[:8]}.csv"
    output_path = os.path.join(config.output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    analysis = analyze_dataframe(df, cleaned_id, output_filename)
    analysis["download_url"] = f"/api/exp1/download/{output_filename}"
    analysis["rows_removed"] = initial_rows - len(df)
    
    return analysis

@router.post("/outliers/handle")
async def handle_outliers(
    dataset_id: str = Form(...),
    column: str = Form(...),
    strategy: str = Form("clip")  # 'clip' (Winsorization) or 'drop'
):
    if dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")
        
    df = DATASET_STORE[dataset_id].copy()
    initial_rows = len(df)
    
    target_cols = [column] if column != "all" else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    for col in target_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) >= 4:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if strategy == "clip":
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                elif strategy == "drop":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
    processed_id = str(uuid.uuid4())
    DATASET_STORE[processed_id] = df
    
    output_filename = f"outliers_processed_{processed_id[:8]}.csv"
    output_path = os.path.join(config.output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    analysis = analyze_dataframe(df, processed_id, output_filename)
    analysis["download_url"] = f"/api/exp1/download/{output_filename}"
    analysis["rows_removed"] = initial_rows - len(df)
    return analysis

@router.post("/scale")
async def scale_features(
    dataset_id: str = Form(...),
    method: str = Form("minmax"),  # 'minmax' or 'standard'
    column: str = Form("all")
):
    if dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")
        
    df = DATASET_STORE[dataset_id].copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [column] if column != "all" and column in numeric_cols else numeric_cols
    
    for col in target_cols:
        series = df[col].dropna()
        if len(series) > 0:
            if method == "minmax":
                min_val = series.min()
                max_val = series.max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    df[col] = df[col].round(4)
            elif method == "standard":
                mean_val = series.mean()
                std_val = series.std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    df[col] = df[col].round(4)
                    
    processed_id = str(uuid.uuid4())
    DATASET_STORE[processed_id] = df
    
    output_filename = f"scaled_dataset_{processed_id[:8]}.csv"
    output_path = os.path.join(config.output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    analysis = analyze_dataframe(df, processed_id, output_filename)
    analysis["download_url"] = f"/api/exp1/download/{output_filename}"
    return analysis

@router.post("/encode")
async def encode_categorical(
    dataset_id: str = Form(...),
    column: str = Form(...),
    method: str = Form("one_hot")  # 'one_hot' or 'label'
):
    if dataset_id not in DATASET_STORE:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")
        
    df = DATASET_STORE[dataset_id].copy()
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found.")
        
    if method == "one_hot":
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=False, dtype=int)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    elif method == "label":
        df[column] = df[column].astype("category").cat.codes
        
    processed_id = str(uuid.uuid4())
    DATASET_STORE[processed_id] = df
    
    output_filename = f"encoded_dataset_{processed_id[:8]}.csv"
    output_path = os.path.join(config.output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    analysis = analyze_dataframe(df, processed_id, output_filename)
    analysis["download_url"] = f"/api/exp1/download/{output_filename}"
    return analysis

@router.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(config.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found.")
    return FileResponse(file_path, media_type="text/csv", filename=filename)
