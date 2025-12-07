import base64
from pathlib import Path
from io import BytesIO

import pytest
import requests
import imageio.v3 as iio
from PIL import Image
import numpy as np


API_URL = "http://localhost:8085"
FRAME_PATH = Path("tests/data/frame.jpg")
VIDEO_CANDIDATES = [
    Path("tests/data/sample.mp4"),
    Path("tests/data/test.mp4"),
]


def _skip_if_no_server():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
    except Exception:
        pytest.skip("API server is not running on localhost:8080")
    if resp.status_code != 200:
        pytest.skip(f"Health check failed: {resp.status_code}")


def _load_frame_b64() -> str:
    if not FRAME_PATH.exists():
        pytest.skip("Add a sample image at tests/data/frame.jpg to run this test")
    frame_bytes = FRAME_PATH.read_bytes()
    return base64.b64encode(frame_bytes).decode("ascii")


def _load_video_frames_b64(num_frames: int = 32):
    video_path = next((p for p in VIDEO_CANDIDATES if p.exists()), None)
    if not video_path:
        pytest.skip("Add a sample video at tests/data/sample.mp4 (or test.mp4) to run video test")

    frames = list(iio.imiter(video_path))
    if not frames:
        pytest.skip("Video has no frames")

    # Sample/pad to num_frames
    if len(frames) >= num_frames:
        indices = (np.linspace(0, len(frames) - 1, num_frames)).astype(int)
        frames = [frames[i] for i in indices]
    else:
        frames = frames + [frames[-1]] * (num_frames - len(frames))

    b64_frames = []
    for frame in frames[:num_frames]:
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64_frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return b64_frames


@pytest.mark.integration
def test_process_endpoint_returns_text():
    _skip_if_no_server()
    frame_b64 = _load_frame_b64()

    payload = {
        "frames": [frame_b64] * 32,
        "count": 32,
    }

    resp = requests.post(f"{API_URL}/process", json=payload, timeout=30)
    print(f"API response: status={resp.status_code}, body={resp.text}")
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert isinstance(data["text"], str)
    assert data["text"].strip() != ""


@pytest.mark.integration
def test_process_endpoint_with_video():
    _skip_if_no_server()
    frames_b64 = _load_video_frames_b64(32)

    payload = {
        "frames": frames_b64,
        "count": len(frames_b64),
    }

    resp = requests.post(f"{API_URL}/process", json=payload, timeout=30)
    print(f"Video API response: status={resp.status_code}, body={resp.text}")
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert isinstance(data["text"], str)
    assert data["text"].strip() != ""
