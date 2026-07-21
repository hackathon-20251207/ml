#!/usr/bin/env python3
"""Replay a video through the deployed upload API and/or live WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import imageio.v3 as iio
import requests
import websockets
from PIL import Image

USER_AGENT = "Sigma-Sign-production-smoke/1.0"


def combined_transcript(final_text: str, draft_text: str) -> str:
    return " ".join(part.strip() for part in (final_text, draft_text) if part.strip())


def websocket_candidate(payload: dict[str, Any]) -> str | None:
    candidate = payload.get("full_text") or payload.get("text")
    return candidate if isinstance(candidate, str) and candidate.strip() else None


def required_string(payload: dict[str, Any], name: str, message_type: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise RuntimeError(
            f"WebSocket {message_type} event has invalid {name}: {value!r}"
        )
    return value


def required_positive_int(payload: dict[str, Any], name: str, message_type: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"WebSocket {message_type} event has invalid {name}: {value!r}"
        )
    return value


@dataclass
class SegmentProgress:
    gestures: list[tuple[int, str]] = field(default_factory=list)
    formatting_seen: bool = False
    transcript_seen: bool = False


class EnhancedTranscriptTracker:
    """Validate and find one complete enhanced two-layer transcript segment."""

    def __init__(self, expected_literal: str | None):
        self.expected_literal = expected_literal.strip() if expected_literal else None
        self.segments: dict[int, SegmentProgress] = {}
        self.last_gesture_sequence = 0

    def observe(self, payload: dict[str, Any]) -> str | None:
        message_type = payload.get("type")
        if message_type not in {"gesture", "formatting", "transcript"}:
            return None

        self._validate_snapshots(payload, message_type)
        segment_id = required_positive_int(payload, "segment_id", message_type)
        progress = self.segments.setdefault(segment_id, SegmentProgress())

        if message_type == "gesture":
            self._observe_gesture(payload, progress)
            self._validate_literal_snapshot(payload, message_type)
            return None
        if message_type == "formatting":
            self._observe_formatting(payload, progress)
            self._validate_literal_snapshot(payload, message_type)
            return None
        self._validate_literal_snapshot(payload, message_type)
        return self._observe_transcript(payload, progress)

    @staticmethod
    def _validate_snapshots(payload: dict[str, Any], message_type: str) -> None:
        final_text = required_string(payload, "final_text", message_type)
        draft_text = required_string(payload, "draft_text", message_type)
        full_text = required_string(payload, "full_text", message_type)
        required_string(payload, "literal_text", message_type)
        expected_full_text = combined_transcript(final_text, draft_text)
        if full_text != expected_full_text:
            raise RuntimeError(
                "WebSocket "
                f"{message_type} event has non-authoritative full_text: "
                f"expected {expected_full_text!r}, got {full_text!r}"
            )

    def _validate_literal_snapshot(
        self, payload: dict[str, Any], message_type: str
    ) -> None:
        observed_tokens = sorted(
            (
                gesture
                for progress in self.segments.values()
                for gesture in progress.gestures
            ),
            key=lambda gesture: gesture[0],
        )
        observed_literal = " ".join(text for _, text in observed_tokens)
        literal_text = required_string(payload, "literal_text", message_type)
        if payload.get("truncated") is True:
            valid = bool(literal_text) and observed_literal.endswith(literal_text)
        else:
            valid = literal_text == observed_literal
        if not valid:
            raise RuntimeError(
                "WebSocket "
                f"{message_type} event has non-authoritative literal_text: "
                f"expected {observed_literal!r}, got {literal_text!r}"
            )

    def _observe_gesture(
        self, payload: dict[str, Any], progress: SegmentProgress
    ) -> None:
        if progress.formatting_seen or progress.transcript_seen:
            raise RuntimeError("WebSocket gesture arrived after its segment was closed")
        if payload.get("status") != "draft":
            raise RuntimeError("WebSocket gesture event must have status='draft'")

        text = required_string(payload, "text", "gesture").strip()
        if not text:
            raise RuntimeError("WebSocket gesture event has empty text")
        sequence = required_positive_int(payload, "sequence", "gesture")
        if sequence <= self.last_gesture_sequence:
            raise RuntimeError(
                "WebSocket gesture sequences are not strictly increasing"
            )
        self.last_gesture_sequence = sequence

        if required_positive_int(payload, "first_sequence", "gesture") != sequence:
            raise RuntimeError(
                "WebSocket gesture first_sequence does not match sequence"
            )
        if required_positive_int(payload, "last_sequence", "gesture") != sequence:
            raise RuntimeError(
                "WebSocket gesture last_sequence does not match sequence"
            )
        if required_positive_int(payload, "token_count", "gesture") != 1:
            raise RuntimeError("WebSocket gesture token_count must equal 1")

        literal_text = required_string(payload, "literal_text", "gesture")
        if text.casefold() not in literal_text.casefold():
            raise RuntimeError("WebSocket gesture text is missing from literal_text")
        progress.gestures.append((sequence, text))

    @staticmethod
    def _observe_formatting(payload: dict[str, Any], progress: SegmentProgress) -> None:
        if not progress.gestures:
            raise RuntimeError("WebSocket formatting arrived before a gesture")
        if progress.formatting_seen or progress.transcript_seen:
            raise RuntimeError(
                "WebSocket formatting event was duplicated or out of order"
            )
        if payload.get("status") != "formatting":
            raise RuntimeError(
                "WebSocket formatting event must have status='formatting'"
            )
        EnhancedTranscriptTracker._validate_segment_metadata(
            payload, progress, "formatting"
        )
        progress.formatting_seen = True

    def _observe_transcript(
        self, payload: dict[str, Any], progress: SegmentProgress
    ) -> str | None:
        if not progress.formatting_seen:
            raise RuntimeError("WebSocket transcript arrived before formatting")
        if progress.transcript_seen:
            raise RuntimeError("WebSocket transcript event was duplicated")
        self._validate_segment_metadata(payload, progress, "transcript")
        progress.transcript_seen = True

        segment_literal = " ".join(text for _, text in progress.gestures)
        literal_text = required_string(payload, "literal_text", "transcript")
        if segment_literal.casefold() not in literal_text.casefold():
            raise RuntimeError(
                "WebSocket transcript literal_text lost its raw gesture segment"
            )

        matches_expected = self.expected_literal is None or (
            self.expected_literal.casefold() in segment_literal.casefold()
        )
        enhanced = payload.get("enhanced")
        if not matches_expected:
            return None
        if enhanced is not True or payload.get("status") != "enhanced":
            raise RuntimeError(
                "WebSocket target segment was not enhanced by the formatter"
            )

        text = required_string(payload, "text", "transcript").strip()
        final_text = required_string(payload, "final_text", "transcript")
        full_text = required_string(payload, "full_text", "transcript")
        if not text or text.casefold() not in final_text.casefold():
            raise RuntimeError(
                "WebSocket enhanced segment text is missing from final_text"
            )
        if not full_text.strip():
            raise RuntimeError("WebSocket enhanced transcript has empty full_text")
        return full_text

    @staticmethod
    def _validate_segment_metadata(
        payload: dict[str, Any], progress: SegmentProgress, message_type: str
    ) -> None:
        sequences = [sequence for sequence, _ in progress.gestures]
        first_sequence = required_positive_int(payload, "first_sequence", message_type)
        last_sequence = required_positive_int(payload, "last_sequence", message_type)
        sequence = required_positive_int(payload, "sequence", message_type)
        token_count = required_positive_int(payload, "token_count", message_type)
        if (
            first_sequence != sequences[0]
            or last_sequence != sequences[-1]
            or sequence != sequences[-1]
            or token_count != len(sequences)
        ):
            raise RuntimeError(
                f"WebSocket {message_type} event has inconsistent segment metadata"
            )


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
    require_enhanced: bool = False,
) -> str:
    transcript = ""
    enhanced_tracker = EnhancedTranscriptTracker(expected) if require_enhanced else None
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
            if not isinstance(payload, dict):
                raise RuntimeError("WebSocket server returned non-object JSON")
            if enhanced_tracker is not None:
                candidate = enhanced_tracker.observe(payload)
                if candidate is not None:
                    transcript = candidate
                    break
                continue

            candidate = websocket_candidate(payload)
            if candidate is not None:
                transcript = candidate
                if expected is None or expected.casefold() in transcript.casefold():
                    break
    if require_enhanced and not transcript.strip():
        raise TimeoutError(
            "WebSocket did not produce the required enhanced transcript segment "
            f"within {timeout_seconds:.0f}s"
        )
    assert_expected(transcript, None if require_enhanced else expected, "WebSocket")
    return transcript


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=normalized_api_url, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("upload", "websocket", "both"), default="both"
    )
    parser.add_argument("--expected")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--require-enhanced",
        action="store_true",
        help=(
            "for WebSocket replay, require an ordered gesture -> formatting -> "
            "enhanced transcript segment; ignored in upload-only mode"
        ),
    )
    args = parser.parse_args(argv)
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
                args.require_enhanced,
            )
        )
        print(f"websocket: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
