"""Torch-free inference API for Vercel (ONNX Runtime + numpy + Pillow)."""
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
META = json.loads((MODEL_DIR / "meta.json").read_text())
HEADS = np.load(MODEL_DIR / "heads.npz")
MEAN = np.array(META["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array(META["std"], dtype=np.float32).reshape(1, 3, 1, 1)
SIZE = tuple(META["image_size"])

_session = None


def session():
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(
            str(MODEL_DIR / "breadfruit.onnx"), opts, providers=["CPUExecutionProvider"]
        )
    return _session


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(SIZE[::-1], Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return ((x - MEAN) / STD).astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def task_result(logits, weights, act, classes):
    probs = softmax(logits)
    idx = int(probs.argmax())
    cam = np.maximum((weights[idx][:, None, None] * act).sum(axis=0), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return {
        "prediction": classes[idx],
        "class_id": idx,
        "confidence": round(float(probs[idx]), 4),
        "grad_cam": np.round(cam, 3).tolist(),  # 7x7 map; the page upsamples it
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "healthy", "runtime": "onnxruntime"}


@router.post("/predict")
@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        x = preprocess(img)
    except Exception as e:
        return JSONResponse({"status": "error", "error": f"Invalid image file: {e}"}, status_code=400)
    act, mat, qual = session().run(None, {"input": x})
    act = act[0]
    return {
        "status": "success",
        "maturity": task_result(mat[0], HEADS["maturity_w"], act, META["maturity_classes"]),
        "quality": task_result(qual[0], HEADS["quality_w"], act, META["quality_classes"]),
    }


app = FastAPI(title="Breadfruit Classification API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
app.include_router(router, prefix="/api")
