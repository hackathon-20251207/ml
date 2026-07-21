#!/usr/bin/env python3
"""Run a deterministic Slovo regression sentinel against the production model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("slovo_golden.json")
DEFAULT_CACHE = ROOT / ".cache" / "model-evaluation"
MODEL_URL = "https://raw.githubusercontent.com/ai-forever/easy_sign/main/S3D.onnx"
MODEL_SHA256 = "860ecb5e5aff91b4709016c2dc4f5744eea53e024f80c0b3b8f0f916f6bdb949"
CLASS_LIST_PATH = ROOT / "offline_inference" / "RSL_class_list.txt"
CLASS_LIST_SHA256 = "390e90884aeac96c03ef6db87754ea62cb15b4a5b58f3659a5a900153e97f672"
USER_AGENT = "Sigma-Sign-model-evaluation/1.0"


class CandidateLike(Protocol):
    text: str


class ResponseLike(Protocol):
    accepted: bool
    candidates: list[CandidateLike]
    confidence: float


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SampleResult:
    id: str
    expected: str
    predicted: str
    confidence: float
    accepted: bool
    top1_correct: bool
    expected_in_top3: bool
    windows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path, expected_sha256: str, expected_bytes: int | None = None
) -> None:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"unexpected size for {path.name}")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {path.name}")


def load_manifest(path: Path) -> tuple[dict[str, object], list[Sample]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported evaluation manifest")
    required = {"archive_url", "license", "samples", "source", "split", "thresholds"}
    if not required.issubset(raw):
        raise ValueError("evaluation manifest is missing required fields")
    if raw["license"] != "CC-BY-SA-4.0" or raw["split"] != "test":
        raise ValueError("evaluation manifest has an unexpected license or split")
    thresholds = raw["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "min_top1",
        "min_top3_any_window",
    }:
        raise ValueError("evaluation manifest has invalid thresholds")
    if not all(
        isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        for value in thresholds.values()
    ):
        raise ValueError("evaluation thresholds must be between 0 and 1")
    rows = raw["samples"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation manifest has no samples")
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "id", "label", "sha256"}:
            raise ValueError("invalid sample entry")
        sample = Sample(
            id=str(row["id"]),
            label=str(row["label"]).strip().casefold(),
            sha256=str(row["sha256"]).lower(),
            bytes=int(row["bytes"]),
        )
        if (
            not sample.id
            or not sample.label
            or len(sample.sha256) != 64
            or sample.bytes <= 0
            or sample.id in seen_ids
            or sample.label in seen_labels
        ):
            raise ValueError("invalid or duplicate sample entry")
        seen_ids.add(sample.id)
        seen_labels.add(sample.label)
        samples.append(sample)
    return raw, samples


def atomic_download(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists():
        try:
            verify_file(target, expected_sha256)
            return
        except RuntimeError:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        verify_file(temporary, expected_sha256)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_samples(
    archive_url: str, split: str, samples: list[Sample], directory: Path
) -> None:
    """Range-download only the pinned entries from the official 16 GB archive."""
    from remotezip import RemoteZip

    directory.mkdir(parents=True, exist_ok=True)
    missing = []
    for sample in samples:
        target = directory / f"{sample.id}.mp4"
        try:
            verify_file(target, sample.sha256, sample.bytes)
        except (FileNotFoundError, RuntimeError):
            target.unlink(missing_ok=True)
            missing.append(sample)
    if not missing:
        return

    with RemoteZip(archive_url, headers={"User-Agent": USER_AGENT}) as archive:
        names = set(archive.namelist())
        for sample in missing:
            member = f"{split}/{sample.id}.mp4"
            if member not in names:
                raise RuntimeError(
                    f"sample is missing from the upstream archive: {member}"
                )
            payload = archive.read(member)
            target = directory / f"{sample.id}.mp4"
            temporary = target.with_suffix(".mp4.download")
            try:
                temporary.write_bytes(payload)
                verify_file(temporary, sample.sha256, sample.bytes)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def decode_video(path: Path) -> list[np.ndarray]:
    frames = []
    for raw_frame in iio.imiter(path, plugin="FFMPEG"):
        frame = np.asarray(raw_frame)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"unexpected frame format in {path.name}")
        frames.append(frame[:, :, :3])
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {path.name}")
    return frames


def window_frames(
    frames: list[np.ndarray], window_size: int = 32, stride: int = 16
) -> list[list[np.ndarray]]:
    if not frames:
        raise ValueError("at least one frame is required")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    windows = []
    for start in range(0, len(frames), stride):
        window = list(frames[start : start + window_size])
        window.extend([window[-1]] * (window_size - len(window)))
        windows.append(window)
        if start + window_size >= len(frames):
            break
    return windows


def score_sample(sample: Sample, responses: list[ResponseLike]) -> SampleResult:
    if not responses:
        raise ValueError("at least one model response is required")
    best = max(responses, key=lambda response: response.confidence)
    predicted = best.candidates[0].text.casefold() if best.candidates else ""
    expected_in_top3 = any(
        sample.label == candidate.text.casefold()
        for response in responses
        for candidate in response.candidates[:3]
    )
    return SampleResult(
        id=sample.id,
        expected=sample.label,
        predicted=predicted,
        confidence=best.confidence,
        accepted=best.accepted,
        top1_correct=predicted == sample.label,
        expected_in_top3=expected_in_top3,
        windows=len(responses),
    )


def configure_model(model_path: Path) -> object:
    os.environ.update(
        {
            "CLASS_LIST_PATH": str(CLASS_LIST_PATH),
            "CLASS_LIST_SHA256": CLASS_LIST_SHA256,
            "MODEL_PATH": str(model_path),
            "MODEL_SHA256": MODEL_SHA256,
            "USE_MOCK": "false",
        }
    )
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("app")


def build_report(
    manifest: dict[str, object],
    results: list[SampleResult],
    min_top1: float,
    min_top3: float,
) -> dict[str, object]:
    count = len(results)
    top1 = sum(result.top1_correct for result in results) / count
    accepted_top1 = (
        sum(result.top1_correct and result.accepted for result in results) / count
    )
    top3 = sum(result.expected_in_top3 for result in results) / count
    return {
        "suite": manifest.get("name", "Slovo fixed regression sentinel"),
        "disclaimer": "Regression sentinel on a fixed subset; not a general accuracy benchmark.",
        "source": manifest["source"],
        "license": manifest["license"],
        "model": {"url": MODEL_URL, "sha256": MODEL_SHA256},
        "class_list_sha256": CLASS_LIST_SHA256,
        "thresholds": {"min_top1": min_top1, "min_top3_any_window": min_top3},
        "metrics": {
            "samples": count,
            "top1": top1,
            "accepted_top1": accepted_top1,
            "top3_any_window": top3,
        },
        "passed": top1 >= min_top1 and top3 >= min_top3,
        "samples": [asdict(result) for result in results],
    }


def write_report(report: dict[str, object], path: Path | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-top1", type=ratio)
    parser.add_argument("--min-top3", type=ratio)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="With --download-only, skip the model and class-list downloads",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, samples = load_manifest(args.manifest)
    if args.videos_only and not args.download_only:
        raise SystemExit("--videos-only requires --download-only")
    if args.sample_id:
        selected_ids = set(args.sample_id)
        samples = [sample for sample in samples if sample.id in selected_ids]
        missing_ids = selected_ids - {sample.id for sample in samples}
        if missing_ids:
            raise SystemExit(f"unknown sample id(s): {', '.join(sorted(missing_ids))}")
    model_path = args.cache_dir / "S3D.onnx"
    video_dir = args.cache_dir / "slovo-test"
    if not args.videos_only:
        verify_file(CLASS_LIST_PATH, CLASS_LIST_SHA256)
        atomic_download(MODEL_URL, model_path, MODEL_SHA256)
    fetch_samples(
        str(manifest["archive_url"]), str(manifest["split"]), samples, video_dir
    )
    if args.download_only:
        print(f"Downloaded and verified {len(samples)} sample(s) in {video_dir}")
        return 0

    model_api = configure_model(model_path)
    results = []
    for index, sample in enumerate(samples, start=1):
        frames = decode_video(video_dir / f"{sample.id}.mp4")
        responses = [model_api.predict(window) for window in window_frames(frames)]
        result = score_sample(sample, responses)
        results.append(result)
        print(
            f"[{index:02d}/{len(samples)}] {sample.label:<12} -> "
            f"{result.predicted or '-':<12} confidence={result.confidence:.3f}",
            file=sys.stderr,
        )

    thresholds = manifest["thresholds"]
    min_top1 = args.min_top1 if args.min_top1 is not None else thresholds["min_top1"]
    min_top3 = (
        args.min_top3
        if args.min_top3 is not None
        else thresholds["min_top3_any_window"]
    )
    report = build_report(manifest, results, min_top1, min_top3)
    write_report(report, args.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
