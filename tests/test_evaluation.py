import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from evaluation.evaluate_slovo import (
    Sample,
    build_report,
    load_manifest,
    score_sample,
    window_frames,
)
from evaluation.replay_stack import normalized_api_url, websocket_url


def candidate(text: str):
    return SimpleNamespace(text=text)


def response(confidence: float, labels: list[str], accepted: bool = True):
    return SimpleNamespace(
        accepted=accepted,
        candidates=[candidate(label) for label in labels],
        confidence=confidence,
    )


def test_window_frames_overlaps_and_pads_last_frame():
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(33)]

    windows = window_frames(frames)

    assert len(windows) == 2
    assert int(windows[0][0][0, 0, 0]) == 0
    assert int(windows[0][-1][0, 0, 0]) == 31
    assert int(windows[1][0][0, 0, 0]) == 16
    assert int(windows[1][-1][0, 0, 0]) == 32


def test_score_uses_highest_confidence_window_and_top3_across_windows():
    sample = Sample(id="sample", label="город", sha256="0" * 64, bytes=1)

    result = score_sample(
        sample,
        [
            response(0.6, ["дом", "город", "улица"]),
            response(0.9, ["маленький", "дом", "улица"]),
        ],
    )

    assert result.predicted == "маленький"
    assert not result.top1_correct
    assert result.expected_in_top3


def test_report_enforces_both_regression_thresholds():
    manifest = {"license": "CC-BY-SA-4.0", "source": "https://example.test"}
    results = [
        score_sample(
            Sample(id=str(index), label="день", sha256="0" * 64, bytes=1),
            [response(0.9, ["день", "утро", "вечер"])],
        )
        for index in range(4)
    ]
    results.append(
        score_sample(
            Sample(id="miss", label="город", sha256="0" * 64, bytes=1),
            [response(0.9, ["дом", "улица", "деревня"])],
        )
    )

    report = build_report(manifest, results, min_top1=0.8, min_top3=0.9)

    assert report["metrics"]["top1"] == pytest.approx(0.8)
    assert not report["passed"]


def test_committed_manifest_keeps_current_regression_floor():
    manifest_path = Path(__file__).parents[1] / "evaluation" / "slovo_golden.json"

    manifest, samples = load_manifest(manifest_path)

    assert len(samples) == 20
    assert manifest["thresholds"] == {
        "min_top1": 0.85,
        "min_top3_any_window": 0.95,
    }


def test_websocket_url_preserves_api_prefix():
    base_url = normalized_api_url("https://hack.eferzo.xyz/api/")

    assert base_url == "https://hack.eferzo.xyz/api"
    assert websocket_url(base_url) == "wss://hack.eferzo.xyz/api/socket"


@pytest.mark.parametrize(
    "value",
    [
        "hack.eferzo.xyz/api",
        "ftp://hack.eferzo.xyz/api",
        "https://user:pass@example.test/api",
    ],
)
def test_base_url_rejects_unsafe_or_relative_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        normalized_api_url(value)
