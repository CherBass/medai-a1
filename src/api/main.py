from pathlib import Path
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = Path("/app/models/model.onnx")

app = FastAPI(title="medai-a1", version="0.1.0")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    exists = MODEL_PATH.exists()
    return {"status": "ready" if exists else "no_model", "model_present": exists}


@app.post("/predict")
def predict() -> dict[str, str]:
    if not MODEL_PATH.exists():
        return {
            "error": "model_not_loaded",
            "hint": "Train and drop model.onnx into models/",
        }
    return {
        "error": "not_implemented",
        "hint": "Real prediction lands with training work",
    }
