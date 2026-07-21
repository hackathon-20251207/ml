#!/usr/bin/env python3
"""Replay a video through the deployed upload API and/or live WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import imageio.v3 as iio
import requests
import websockets
from PIL import Image

USER_AGENT = "Sigma-Sign-production-smoke/1.0"


def normalized_api_url(value: str) -> str:
    parsed = urlparse(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            "base URL cannot contain credentials, query or fragment"
        )
    return urlunparse(parsed)


def websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/socket"
    return urlunparse(parsed._replace(scheme=scheme, path=path))


def assert_expected(transcript: str, expected: str | None, mode: str) -> None:
    if not transcript.strip():
        raise RuntimeError(f"{mode} smoke returned an empty transcript")
    if expected and expected.casefold() not in transcript.casefold():
        raise RuntimeError(
            f"{mode} transcript does not contain {expected!r}: {transcript!r}"
        )


def replay_upload(
    base_url: str, video: Path, expected: str | None, timeout_seconds: float
) -> str:
    deadline = time.monotonic() + timeout_seconds
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    with video.open("rb") as source:
        response = session.post(
            f"{base_url}/upload",
            files={"video": (video.name, source, "video/mp4")},
            timeout=(10, 60),
        )
    response.raise_for_status()
    job_id = response.json().get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("upload API returned no job id")

    while time.monotonic() < deadline:
        status_response = session.get(f"{base_url}/job/{job_id}", timeout=(10, 30))
        status_response.raise_for_status()
        job = status_response.json()
        status = job.get("status")
        if status == "completed":
            transcript = str(job.get("full_text") or job.get("text") or "")
            assert_expected(transcript, expected, "upload")
            return transcript
        if status == "failed":
            raise RuntimeError(
                f"upload job failed: {job.get('error') or 'unknown error'}"
            )
        retry_after = float(status_response.headers.get("Retry-After", "1"))
        time.sleep(min(max(retry_after, 0.25), 2.0))
    raise TimeoutError(f"upload job did not finish within {timeout_seconds:.0f}s")


def jpeg_frames(video: Path, max_side: int = 640, quality: int = 80):
    for raw_frame in iio.imiter(video, plugin="FFMPEG"):
        image = Image.fromarray(raw_frame[:, :, :3]).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        yield buffer.getvalue()


async def replay_websocket(
    base_url: str,
    video: Path,
    expected: str | None,
    timeout_seconds: float,
    frames_per_second: float,
) -> str:
    transcript = ""
    socket_url = websocket_url(base_url)
    ssl_context = (
        ssl.create_default_context(cafile=requests.certs.where())
        if socket_url.startswith("wss://")
        else None
    )
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    async with websockets.connect(
        socket_url,
        additional_headers={"User-Agent": USER_AGENT},
        max_size=1024 * 1024,
        open_timeout=15,
        close_timeout=5,
        ssl=ssl_context,
    ) as socket:
        for frame in jpeg_frames(video):
            await socket.send(frame)
            await asyncio.sleep(1 / frames_per_second)

        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                message = await asyncio.wait_for(
                    socket.recv(), timeout=min(remaining, 5.0)
                )
            except TimeoutError:
                continue
            if not isinstance(message, str):
                continue
            payload = json.loads(message)
            candidate = payload.get("full_text") or payload.get("text")
            if isinstance(candidate, str) and candidate.strip():
                transcript = candidate
                if expected is None or expected.casefold() in transcript.casefold():
                    break
    assert_expected(transcript, expected, "WebSocket")
    return transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=normalized_api_url, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("upload", "websocket", "both"), default="both"
    )
    parser.add_argument("--expected")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("--video must point to a readable file")
    if args.timeout <= 0 or args.fps <= 0 or args.fps > 60:
        parser.error("--timeout and --fps must be positive; --fps cannot exceed 60")
    return args


def main() -> int:
    args = parse_args()
    if args.mode in {"upload", "both"}:
        transcript = replay_upload(
            args.base_url, args.video, args.expected, args.timeout
        )
        print(f"upload: {transcript}")
    if args.mode in {"websocket", "both"}:
        transcript = asyncio.run(
            replay_websocket(
                args.base_url,
                args.video,
                args.expected,
                args.timeout,
                args.fps,
            )
        )
        print(f"websocket: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
