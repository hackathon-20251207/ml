import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List

from botocore.config import Config
import boto3
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign-api")

# --- Configuration ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/mvit32-2.onnx"))
CLASS_LIST_PATH = Path(os.getenv("CLASS_LIST_PATH", "artifacts/RSL_class_list.txt"))
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "32"))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224"))
USE_MOCK = os.getenv("USE_MOCK", "false").lower() in {"1", "true", "yes"}
FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD", "false").lower() in {"1", "true", "yes"}

S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_KEY = os.getenv("MODEL_KEY", "mvit32-2.onnx")
CLASS_LIST_KEY = os.getenv("CLASS_LIST_KEY", "RSL_class_list.txt")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")  # optional: for S3-compatible storage


class ProcessRequest(BaseModel):
    frames: List[str] = Field(..., description="Base64-encoded image bytes")
    count: int = Field(..., description="Number of frames sent (len(frames))")


class ProcessResponse(BaseModel):
    text: str


def ensure_artifacts():
    """Download model and class list from S3 if they are missing locally."""
    if USE_MOCK:
        log.info("USE_MOCK enabled: skipping artifact download")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    to_download = []
    if FORCE_DOWNLOAD:
        to_download = [(MODEL_KEY, MODEL_PATH), (CLASS_LIST_KEY, CLASS_LIST_PATH)]
    else:
        if not MODEL_PATH.exists():
            to_download.append((MODEL_KEY, MODEL_PATH))
        if not CLASS_LIST_PATH.exists():
            to_download.append((CLASS_LIST_KEY, CLASS_LIST_PATH))

    if not to_download:
        log.info("Artifacts already present locally; skipping download", extra={"path": str(MODEL_PATH)})
        return

    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is not set; cannot download artifacts.")

    log.info("Downloading artifacts from S3", extra={"bucket": S3_BUCKET, "items": [str(t[1]) for t in to_download]})
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    for key, target in to_download:
        s3.download_file(S3_BUCKET, key, str(target))
        log.info("Downloaded %s to %s", key, target)


def load_class_names(path: Path):
    with path.open("r", encoding="utf-8") as f:
        class_names = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                class_names[int(parts[0])] = parts[1]
        return class_names


def load_model(model_path: Path):
    # CPU provider is enough here; adjust if GPU is available.
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def decode_frames_b64(frames_b64: List[str]) -> List[np.ndarray]:
    decoded = []
    for idx, frame_b64 in enumerate(frames_b64):
        try:
            raw = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"frame {idx}: invalid image/base64") from exc

        decoded.append(np.asarray(img))
    return decoded


def prepare_clip(frames: List[np.ndarray], num_frames: int = NUM_FRAMES, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize/normalize frames and shape them for the model."""
    if not frames:
        raise ValueError("no frames provided")

    # Select or pad frames to match num_frames
    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames + [frames[-1]] * (num_frames - len(frames))

    processed = []
    for frame in selected:
        pil_img = Image.fromarray(frame)
        resized = pil_img.resize((size, size), Image.BILINEAR)
        processed.append(np.asarray(resized).astype(np.float32) / 255.0)

    clip = np.stack(processed)  # (T, H, W, C)
    clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
    clip = np.expand_dims(clip, axis=0)  # (1, C, T, H, W)
    return clip


def predict(frames: List[np.ndarray]) -> str:
    clip = prepare_clip(frames)

    # Adjust rank to match model expectation
    if MODEL_INPUT_RANK == 6:
        clip = np.expand_dims(clip, axis=1)  # (1, 1, C, T, H, W)
    elif MODEL_INPUT_RANK == 5:
        # Already (1, C, T, H, W)
        pass
    else:
        raise ValueError(f"Unsupported model input rank: {MODEL_INPUT_RANK}")

    output = MODEL.run(None, {MODEL_INPUT_NAME: clip})[0]  # (1, num_classes)
    predicted_class_id = int(np.argmax(output[0]))
    return CLASS_NAMES.get(predicted_class_id, f"Unknown gesture (ID: {predicted_class_id})")


# Initialize artifacts and model at startup
if USE_MOCK:
    CLASS_NAMES = {}
    MODEL = None
    MODEL_INPUT_NAME = ""
    MODEL_INPUT_RANK = 0
    log.info('Mock mode enabled: model is not loaded; responses will be "(Это МОК)"')
else:
    ensure_artifacts()
    CLASS_NAMES = load_class_names(CLASS_LIST_PATH)
    MODEL = load_model(MODEL_PATH)
    MODEL_INPUT_NAME = MODEL.get_inputs()[0].name
    MODEL_INPUT_RANK = len(MODEL.get_inputs()[0].shape)
    log.info(
        "Model and class list loaded",
        extra={"classes": len(CLASS_NAMES), "input_name": MODEL_INPUT_NAME, "input_rank": MODEL_INPUT_RANK},
    )


app = FastAPI(title="Sign Language Inference API")


@app.get("/health")
def health():
    return "OK"


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if req.count != len(req.frames):
        raise HTTPException(status_code=400, detail="count does not match frames length")

    try:
        decoded_frames = decode_frames_b64(req.frames)
        text = "(Это МОК)" if USE_MOCK else predict(decoded_frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety
        log.exception("failed to process frames")
        raise HTTPException(status_code=500, detail="internal error") from exc

    return ProcessResponse(text=text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
