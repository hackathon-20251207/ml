import base64
import asyncio
import importlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError


@pytest.fixture(scope="module")
def app_module():
    # Unit tests exercise pure preprocessing/postprocessing without loading ONNX.
    import os

    os.environ["USE_MOCK"] = "true"
    return importlib.import_module("app")


def jpeg_b64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_stable_softmax_is_finite_and_normalized(app_module):
    probabilities = app_module.stable_softmax(np.array([10_000.0, 9_999.0, -10_000.0]))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_stable_softmax_rejects_non_finite_logits(app_module):
    with pytest.raises(RuntimeError, match="non-finite"):
        app_module.stable_softmax(np.array([1.0, np.nan]))


def test_artifact_source_includes_storage_location(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "S3_ENDPOINT_URL", "https://objects.example.test")
    first = app_module.artifact_source("bucket-a", "artifacts/model.onnx")
    second = app_module.artifact_source("bucket-b", "artifacts/model.onnx")
    assert first != second
    assert "objects.example.test" in first


def test_prediction_accepts_confident_class(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(app_module, "MIN_MARGIN", 0.1)
    result = app_module.select_prediction(
        np.array([5.0, 0.0, -1.0]), {0: "привет", 1: "нет", 2: "да"}
    )
    assert result.accepted is True
    assert result.text == "привет"
    assert result.class_id == 0
    assert result.confidence > 0.9


def test_prediction_rejects_ambiguous_window(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MIN_CONFIDENCE", 0.2)
    monkeypatch.setattr(app_module, "MIN_MARGIN", 0.1)
    result = app_module.select_prediction(
        np.array([1.0, 0.99, -2.0]), {0: "один", 1: "два", 2: "три"}
    )
    assert result.accepted is False
    assert result.text == ""
    assert result.reason == "ambiguous"


def test_margin_uses_runner_up_when_top_k_is_one(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "TOP_K", 1)
    monkeypatch.setattr(app_module, "MIN_CONFIDENCE", 0.2)
    monkeypatch.setattr(app_module, "MIN_MARGIN", 0.2)
    result = app_module.select_prediction(np.array([1.0, 0.9]), {0: "один", 1: "два"})
    assert len(result.candidates) == 1
    assert result.accepted is False
    assert result.reason == "ambiguous"


def test_prediction_rejects_no_gesture(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "NO_GESTURE_IDS", {14})
    monkeypatch.setattr(app_module, "NO_GESTURE_LABELS", {"no"})
    logits = np.full(15, -5.0)
    logits[14] = 5.0
    result = app_module.select_prediction(
        logits, {index: ("no" if index == 14 else str(index)) for index in range(15)}
    )
    assert result.accepted is False
    assert result.reason == "no_gesture"
    assert result.text == ""


def test_request_requires_exact_frame_count(app_module):
    frame = jpeg_b64()
    with pytest.raises(ValidationError):
        app_module.ProcessRequest(frames=[frame] * 31, count=31)


def test_decode_rejects_invalid_base64(app_module):
    with pytest.raises(ValueError, match="frame 0"):
        app_module.decode_frames_b64(["not valid base64!!!"])


def test_decode_accepts_small_jpeg(app_module):
    decoded = app_module.decode_frames_b64([jpeg_b64()])
    assert len(decoded) == 1
    assert decoded[0].shape == (8, 8, 3)


def test_busy_response_includes_retry_after(app_module, monkeypatch):
    class BusySlot:
        def acquire(self, timeout):
            return False

    monkeypatch.setattr(app_module, "INFERENCE_SLOT", BusySlot())
    request = app_module.ProcessRequest(
        frames=[jpeg_b64()] * app_module.NUM_FRAMES, count=app_module.NUM_FRAMES
    )
    with pytest.raises(app_module.HTTPException) as caught:
        app_module.process(request)
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "1"}


def test_body_limit_rejects_chunked_request_without_content_length(app_module):
    reached_downstream = False
    sent_messages = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"a" * 60, "more_body": True},
            {"type": "http.request", "body": b"b" * 60, "more_body": False},
        ]
    )

    async def downstream(scope, receive, send):
        nonlocal reached_downstream
        reached_downstream = True

    async def receive():
        return next(incoming)

    async def send(message):
        sent_messages.append(message)

    middleware = app_module.RequestBodyLimitMiddleware(downstream, max_body_size=100)
    scope = {"type": "http", "method": "POST", "headers": []}
    asyncio.run(middleware(scope, receive, send))

    assert reached_downstream is False
    assert sent_messages[0]["type"] == "http.response.start"
    assert sent_messages[0]["status"] == 413
