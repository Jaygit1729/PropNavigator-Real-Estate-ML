# src/model_building/mb_error_analysis.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
from src.logger_utils import setup_logger


logger = setup_logger(__name__, "logs/mb_error_analysis.log")


def _segment_metrics(y_true, y_pred, segment_labels, segment_name):
    """Computes per-segment MAPE, MAE, count, and median error."""
    results = []

    # Iterate segments in a sensible order and skip missing labels.
    # Ordered categoricals (e.g. price brackets) keep their natural order;
    # plain labels are sorted. Dropping NaN avoids comparing float vs str.
    labels = segment_labels.dropna()
    if isinstance(labels.dtype, pd.CategoricalDtype) and labels.cat.ordered:
        seg_values = [c for c in labels.cat.categories if (labels == c).any()]
    else:
        seg_values = sorted(labels.unique())

    for seg in seg_values:
        mask = segment_labels == seg
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        results.append({
            "segment": seg,
            "segment_type": segment_name,
            "count": int(mask.sum()),
            "mape_%": round(mean_absolute_percentage_error(yt, yp) * 100, 2),
            "mae_cr": round(mean_absolute_error(yt, yp), 3),
            "median_ae_cr": round(median_absolute_error(yt, yp), 3),
            "r2": round(r2_score(yt, yp), 4),
        })
    return pd.DataFrame(results)


def run_error_analysis(
    pipeline,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    X_test_raw: pd.DataFrame = None,
    output_dir: str = "data/error_analysis"
):
    """
    Comprehensive error analysis on test set predictions.

    Produces:
    1. residuals.csv         — per-row residuals with features
    2. segment_metrics.csv   — MAPE/MAE broken down by property_type,
                               price bracket, and sector
    3. worst_predictions.csv — the 30 worst predictions by absolute error
    4. error_summary.csv     — overall metrics + quantiles
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    y_pred_log = pipeline.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = np.abs(residuals / y_true) * 100

    # 1. Per-row residual DataFrame
    residual_df = X_test.copy()
    residual_df["y_true_cr"] = y_true.values
    residual_df["y_pred_cr"] = np.round(y_pred, 3)
    residual_df["residual_cr"] = np.round(residuals.values, 3)
    residual_df["abs_error_cr"] = np.round(abs_errors.values, 3)
    residual_df["pct_error"] = np.round(pct_errors.values, 2)
    residual_df.to_csv(f"{output_dir}/residuals.csv", index=False)

    # 2. Segmented metrics
    segments = []

    price_brackets = pd.cut(
        y_true, bins=[0, 1, 3, 5, 10, np.inf],   # top bin is open so luxury (>100 Cr) is included
        labels=["<1 Cr", "1-3 Cr", "3-5 Cr", "5-10 Cr", "10+ Cr"]
    )
    segments.append(
        _segment_metrics(y_true, y_pred, price_brackets, "price_bracket")
    )

    if "property_type" in X_test.columns:
        segments.append(
            _segment_metrics(
                y_true, y_pred, X_test["property_type"], "property_type"
            )
        )

    if "sector" in X_test.columns:
        segments.append(
            _segment_metrics(y_true, y_pred, X_test["sector"], "sector")
        )

    segment_df = pd.concat(segments, ignore_index=True)
    segment_df.to_csv(f"{output_dir}/segment_metrics.csv", index=False)
    logger.info(
        f"Segmented metrics computed across "
        f"{segment_df['segment_type'].nunique()} dimensions."
    )

    # 3. Worst predictions
    worst_idx = np.argsort(abs_errors.values)[-30:][::-1]
    worst_df = residual_df.iloc[worst_idx]
    worst_df.to_csv(f"{output_dir}/worst_predictions.csv", index=False)

    # 4. Summary statistics
    overall_mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_r2 = r2_score(y_true, y_pred)
    mdape = float(np.median(pct_errors))
    p90_error = float(np.percentile(pct_errors, 90))
    p95_error = float(np.percentile(pct_errors, 95))
    max_error = float(pct_errors.max())

    summary = pd.DataFrame([{
        "metric": "MAPE (%)", "value": round(overall_mape, 2)
    }, {
        "metric": "MdAPE (%)", "value": round(mdape, 2)
    }, {
        "metric": "MAE (Cr)", "value": round(overall_mae, 3)
    }, {
        "metric": "R²", "value": round(overall_r2, 4)
    }, {
        "metric": "P90 Error (%)", "value": round(p90_error, 2)
    }, {
        "metric": "P95 Error (%)", "value": round(p95_error, 2)
    }, {
        "metric": "Max Error (%)", "value": round(max_error, 2)
    }, {
        "metric": "Test Count", "value": int(len(y_true))
    }])
    summary.to_csv(f"{output_dir}/error_summary.csv", index=False)

    logger.info(
        f"Error analysis complete — MAPE: {overall_mape:.2f}%, "
        f"MdAPE: {mdape:.2f}%, P90: {p90_error:.2f}%"
    )

    return {
        "residual_df": residual_df,
        "segment_df": segment_df,
        "summary": summary
    }
