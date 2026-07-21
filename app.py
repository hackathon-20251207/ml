import base64
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import onnxruntime as ort
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sign-api")


def env_int(name: str, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_sha256(name: str) -> str:
    value = os.getenv(name, "").strip().lower()
    if value and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value


# Model and request limits. The ML service is internal, but it still treats the
# backend payload as untrusted input so a malformed frame cannot exhaust RAM.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/mvit32-2.onnx"))
CLASS_LIST_PATH = Path(os.getenv("CLASS_LIST_PATH", "artifacts/RSL_class_list.txt"))
NUM_FRAMES = env_int("NUM_FRAMES", 32)
INPUT_SIZE = env_int("INPUT_SIZE", 224)
MIN_CONFIDENCE = env_float("MIN_CONFIDENCE", 0.5)
MIN_MARGIN = env_float("MIN_MARGIN", 0.1)
TOP_K = env_int("TOP_K", 3, maximum=10)
MAX_FRAME_BYTES = env_int("MAX_FRAME_BYTES", 512 * 1024)
MAX_FRAME_BASE64_CHARS = ((MAX_FRAME_BYTES + 2) // 3) * 4
MAX_IMAGE_SIDE = env_int("MAX_IMAGE_SIDE", 2048)
MAX_IMAGE_PIXELS = env_int("MAX_IMAGE_PIXELS", 2_000_000)
MAX_REQUEST_BYTES = env_int(
    "MAX_REQUEST_BYTES", NUM_FRAMES * (MAX_FRAME_BASE64_CHARS + 8) + 4096
)
INFERENCE_WAIT_SECONDS = env_float("INFERENCE_WAIT_SECONDS", 0.25, maximum=30.0)
ONNX_THREADS = env_int("ONNX_THREADS", min(2, os.cpu_count() or 1))

USE_MOCK = os.getenv("USE_MOCK", "false").lower() in {"1", "true", "yes"}
FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD", "false").lower() in {"1", "true", "yes"}

S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_KEY = os.getenv("MODEL_KEY", "mvit32-2.onnx")
CLASS_LIST_KEY = os.getenv("CLASS_LIST_KEY", "RSL_class_list.txt")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
MODEL_SHA256 = env_sha256("MODEL_SHA256")
CLASS_LIST_SHA256 = env_sha256("CLASS_LIST_SHA256")

NO_GESTURE_LABELS = {
    label.strip().casefold()
    for label in os.getenv("NO_GESTURE_LABELS", "no").split(",")
    if label.strip()
}
NO_GESTURE_IDS = {
    int(value.strip())
    for value in os.getenv("NO_GESTURE_IDS", "14").split(",")
    if value.strip()
}

INFERENCE_SLOT = threading.BoundedSemaphore(value=1)


class ProcessRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        min_length=NUM_FRAMES,
        max_length=NUM_FRAMES,
        description="Exactly NUM_FRAMES base64-encoded images",
    )
    count: int = Field(..., ge=NUM_FRAMES, le=NUM_FRAMES)

    @model_validator(mode="after")
    def validate_frame_count(self) -> "ProcessRequest":
        if self.count != len(self.frames):
            raise ValueError("count does not match frames length")
        for index, frame in enumerate(self.frames):
            if not frame or len(frame) > MAX_FRAME_BASE64_CHARS:
                raise ValueError(f"frame {index}: encoded frame is too large or empty")
        return self


class Candidate(BaseModel):
    class_id: int
    text: str
    confidence: float


class ProcessResponse(BaseModel):
    text: str = ""
    class_id: Optional[int] = None
    confidence: float = 0.0
    candidates: list[Candidate] = Field(default_factory=list)
    accepted: bool = False
    reason: Optional[str] = None


class RequestBodyLimitMiddleware:
    """Enforce the request limit even for HTTP/1.1 chunked bodies."""

    def __init__(self, app, max_body_size: int):
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_size:
                    await Response("request body too large", status_code=413)(scope, receive, send)
                    return
            except ValueError:
                await Response("invalid content-length", status_code=400)(scope, receive, send)
                return

        received = 0
        messages = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            received += len(message.get("body", b""))
            if received > self.max_body_size:
                await Response("request body too large", status_code=413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        message_iterator = iter(messages)

        async def replay_receive():
            return next(
                message_iterator,
                {"type": "http.request", "body": b"", "more_body": False},
            )

        await self.app(scope, replay_receive, send)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path.name}")


def download_artifact(s3, bucket: str, key: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".download", delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        s3.download_file(bucket, key, str(temp_path))
        verify_checksum(temp_path, expected_sha256)
        temp_path.replace(target)
        write_source_marker(target, artifact_source(bucket, key))
        log.info("Downloaded artifact %s", target.name)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def source_marker_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.source")


def artifact_source(bucket: str, key: str) -> str:
    return json.dumps(
        {"bucket": bucket, "endpoint": S3_ENDPOINT_URL or "", "key": key},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_source_marker(target: Path, key: str) -> None:
    marker = source_marker_path(target)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".source",
            delete=False,
        ) as temporary:
            temporary.write(key)
            temp_path = Path(temporary.name)
        temp_path.replace(marker)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def ensure_artifacts() -> None:
    """Download missing or explicitly refreshed artifacts atomically."""
    if USE_MOCK:
        log.info("USE_MOCK enabled: skipping artifact download")
        return

    artifacts = [
        (MODEL_KEY, MODEL_PATH, MODEL_SHA256),
        (CLASS_LIST_KEY, CLASS_LIST_PATH, CLASS_LIST_SHA256),
    ]
    to_download: list[tuple[str, Path, str]] = []
    for key, target, checksum in artifacts:
        if target.exists() and not FORCE_DOWNLOAD:
            try:
                verify_checksum(target, checksum)
                marker_matches = (
                    source_marker_path(target).read_text(encoding="utf-8").strip()
                    == artifact_source(S3_BUCKET, key)
                    if source_marker_path(target).exists()
                    else False
                )
                if checksum or marker_matches:
                    continue
                log.warning("Cached artifact %s has no matching source marker", target.name)
            except (OSError, UnicodeError, RuntimeError):
                log.warning("Cached artifact %s failed cache validation", target.name)
        to_download.append((key, target, checksum))

    if not to_download:
        log.info("Artifacts already present and valid")
        return
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is not set; cannot download artifacts")

    s3 = boto3.session.Session().client(
        "s3",
        region_name=os.getenv("AWS_REGION"),
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=90,
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    for key, target, checksum in to_download:
        download_artifact(s3, S3_BUCKET, key, target, checksum)


def load_class_names(path: Path) -> dict[int, str]:
    class_names: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise RuntimeError(f"invalid class list line {line_number}")
            try:
                class_id = int(parts[0])
            except ValueError as exc:
                raise RuntimeError(f"invalid class id on line {line_number}") from exc
            if class_id in class_names:
                raise RuntimeError(f"duplicate class id {class_id}")
            class_names[class_id] = parts[1].strip()
    if not class_names:
        raise RuntimeError("class list is empty")
    return class_names


def validate_model_contract(model, class_names: dict[int, str]) -> tuple[str, int]:
    model_input = model.get_inputs()[0]
    model_output = model.get_outputs()[0]
    rank = len(model_input.shape)
    if rank not in {5, 6}:
        raise RuntimeError(f"unsupported model input rank: {rank}")
    if model_input.type != "tensor(float)":
        raise RuntimeError(f"unsupported model input type: {model_input.type}")

    expected_dimensions = {
        0: 1,
        1: 3,
        2: NUM_FRAMES,
        rank - 2: INPUT_SIZE,
        rank - 1: INPUT_SIZE,
    }
    if rank == 6:
        expected_dimensions[3] = 1
    for index, expected in expected_dimensions.items():
        actual = model_input.shape[index]
        if isinstance(actual, int) and actual != expected:
            raise RuntimeError(
                f"model input dimension {index} is {actual}, configured value is {expected}"
            )

    if model_output.shape and isinstance(model_output.shape[0], int) and model_output.shape[0] != 1:
        raise RuntimeError(f"model output batch dimension is {model_output.shape[0]}, expected 1")

    output_classes = model_output.shape[-1]
    if isinstance(output_classes, int):
        if output_classes != len(class_names):
            raise RuntimeError(
                f"model outputs {output_classes} classes, class list contains {len(class_names)}"
            )
        if set(class_names) != set(range(output_classes)):
            raise RuntimeError("class ids must be contiguous and match model output indexes")
    return model_input.name, rank


def load_model(model_path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = ONNX_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def decode_frames_b64(frames_b64: list[str]) -> list[np.ndarray]:
    decoded: list[np.ndarray] = []
    for index, frame_b64 in enumerate(frames_b64):
        try:
            raw = base64.b64decode(frame_b64, validate=True)
            if not raw or len(raw) > MAX_FRAME_BYTES:
                raise ValueError("decoded frame is too large or empty")

            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as probe:
                    if probe.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError("unsupported image format")
                    width, height = probe.size
                    if (
                        width <= 0
                        or height <= 0
                        or width > MAX_IMAGE_SIDE
                        or height > MAX_IMAGE_SIDE
                        or width * height > MAX_IMAGE_PIXELS
                    ):
                        raise ValueError("image dimensions are too large")
                    probe.verify()

                with Image.open(BytesIO(raw)) as image:
                    image.load()
                    decoded.append(np.asarray(image.convert("RGB")))
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError(f"frame {index}: invalid image/base64") from exc
    return decoded


def prepare_clip(
    frames: list[np.ndarray], num_frames: int = NUM_FRAMES, size: int = INPUT_SIZE
) -> np.ndarray:
    if len(frames) != num_frames:
        raise ValueError(f"exactly {num_frames} frames are required")

    processed: list[np.ndarray] = []
    for frame in frames:
        resized = Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)
        processed.append(np.asarray(resized, dtype=np.float32) / 255.0)

    clip = np.stack(processed)
    clip = np.transpose(clip, (3, 0, 1, 2))
    return np.expand_dims(clip, axis=0)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("model returned empty or non-finite logits")
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def select_prediction(logits: np.ndarray, class_names: dict[int, str]) -> ProcessResponse:
    probabilities = stable_softmax(logits)
    top_count = min(TOP_K, len(probabilities))
    decision_count = min(max(top_count, 2), len(probabilities))
    top_ids = np.argsort(probabilities)[-decision_count:][::-1]
    ranked_candidates = [
        Candidate(
            class_id=int(class_id),
            text=class_names.get(int(class_id), ""),
            confidence=float(probabilities[class_id]),
        )
        for class_id in top_ids
    ]
    candidates = ranked_candidates[:top_count]
    best = candidates[0]
    runner_up = ranked_candidates[1].confidence if len(ranked_candidates) > 1 else 0.0
    margin = best.confidence - runner_up

    reason: Optional[str] = None
    if not best.text:
        reason = "unknown_class"
    elif best.class_id in NO_GESTURE_IDS or best.text.casefold() in NO_GESTURE_LABELS:
        reason = "no_gesture"
    elif best.confidence < MIN_CONFIDENCE:
        reason = "low_confidence"
    elif margin < MIN_MARGIN:
        reason = "ambiguous"

    return ProcessResponse(
        text=best.text if reason is None else "",
        class_id=best.class_id,
        confidence=best.confidence,
        candidates=candidates,
        accepted=reason is None,
        reason=reason,
    )


def predict(frames: list[np.ndarray]) -> ProcessResponse:
    clip = prepare_clip(frames)
    if MODEL_INPUT_RANK == 6:
        clip = np.expand_dims(clip, axis=3)  # (1, C, T, 1, H, W)
    elif MODEL_INPUT_RANK != 5:
        raise ValueError(f"unsupported model input rank: {MODEL_INPUT_RANK}")

    output = MODEL.run(None, {MODEL_INPUT_NAME: clip})[0]
    return select_prediction(output[0], CLASS_NAMES)


if USE_MOCK:
    CLASS_NAMES: dict[int, str] = {}
    MODEL = None
    MODEL_INPUT_NAME = ""
    MODEL_INPUT_RANK = 0
    log.info("Mock mode enabled: model is not loaded")
else:
    ensure_artifacts()
    CLASS_NAMES = load_class_names(CLASS_LIST_PATH)
    MODEL = load_model(MODEL_PATH)
    MODEL_INPUT_NAME, MODEL_INPUT_RANK = validate_model_contract(MODEL, CLASS_NAMES)
    log.info(
        "Model loaded: classes=%d input=%s rank=%d threads=%d",
        len(CLASS_NAMES),
        MODEL_INPUT_NAME,
        MODEL_INPUT_RANK,
        ONNX_THREADS,
    )


app = FastAPI(title="Sign Language Inference API")
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_REQUEST_BYTES)


@app.get("/health")
def health():
    return "OK"


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if not INFERENCE_SLOT.acquire(timeout=INFERENCE_WAIT_SECONDS):
        raise HTTPException(
            status_code=503,
            detail="inference service is busy",
            headers={"Retry-After": "1"},
        )

    try:
        decoded_frames = decode_frames_b64(req.frames)
        if USE_MOCK:
            return ProcessResponse(
                text="(Это МОК)",
                class_id=0,
                confidence=1.0,
                candidates=[Candidate(class_id=0, text="(Это МОК)", confidence=1.0)],
                accepted=True,
            )
        return predict(decoded_frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime safety
        log.exception("failed to process frames")
        raise HTTPException(status_code=500, detail="internal error") from exc
    finally:
        INFERENCE_SLOT.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8085")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=1,
        limit_concurrency=4,
        backlog=16,
    )
