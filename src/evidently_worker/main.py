import math
import time
import traceback

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import Gauge, start_http_server

DRIFT_SHARE = Gauge(
    "medai_data_drift_share",
    "Share of features detected as drifted (0.0-1.0)",
)
DRIFT_DETECTED = Gauge(
    "medai_data_drift_detected",
    "Whether any drift was detected (1) or not (0)",
)
ITERATIONS = Gauge(
    "medai_drift_iterations_total",
    "Number of drift-detection iterations completed",
)


def synth_reference(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.uniform(0, 10, n),
            "feature_c": rng.exponential(2, n),
        }
    )


def synth_current(step: int, n: int = 500) -> pd.DataFrame:
    """Drift factor oscillates: sin(step/3)*3 — period ~19 steps ~10 min."""
    rng = np.random.default_rng()
    drift = math.sin(step / 3) * 3
    return pd.DataFrame(
        {
            "feature_a": rng.normal(drift, 1, n),
            "feature_b": rng.uniform(0, 10 + abs(drift), n),
            "feature_c": rng.exponential(2, n),
        }
    )


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()
    metric_result = result["metrics"][0]["result"]
    return {
        "share": float(metric_result.get("share_of_drifted_columns", 0.0)),
        "detected": int(bool(metric_result.get("dataset_drift", False))),
    }


def main() -> None:
    start_http_server(8001)
    reference = synth_reference()
    print("evidently-worker: metrics on :8001/metrics", flush=True)

    step = 0
    while True:
        try:
            current = synth_current(step)
            stats = compute_drift(reference, current)
            DRIFT_SHARE.set(stats["share"])
            DRIFT_DETECTED.set(stats["detected"])
            ITERATIONS.inc()
            print(
                f"[step={step}] share={stats['share']:.3f} detected={stats['detected']}",
                flush=True,
            )
        except Exception:
            print(f"[step={step}] error — see traceback", flush=True)
            traceback.print_exc()
        step += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
