import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.evaluate_slovo import (
    Sample,
    build_report,
    load_manifest,
    score_sample,
    window_frames,
)
from evaluation.replay_stack import (
    EnhancedTranscriptTracker,
    normalized_api_url,
    parse_args,
    websocket_candidate,
    websocket_url,
)


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


def test_enhanced_tracker_waits_for_ordered_authoritative_transcript():
    tracker = EnhancedTranscriptTracker("работать")

    assert (
        tracker.observe(
            live_event(
                "gesture",
                text="я",
                final_text="",
                draft_text="я",
                literal_text="я",
                sequence=1,
                first_sequence=1,
                last_sequence=1,
                token_count=1,
                status="draft",
            )
        )
        is None
    )
    assert (
        tracker.observe(
            live_event(
                "gesture",
                text="работать",
                final_text="",
                draft_text="я работать",
                literal_text="я работать",
                sequence=2,
                first_sequence=2,
                last_sequence=2,
                token_count=1,
                status="draft",
            )
        )
        is None
    )
    assert (
        tracker.observe(
            live_event(
                "formatting",
                text="",
                final_text="",
                draft_text="я работать",
                literal_text="я работать",
                sequence=2,
                first_sequence=1,
                last_sequence=2,
                token_count=2,
                status="formatting",
            )
        )
        is None
    )

    # A newer gesture can remain as a draft while the first segment is formatted.
    assert (
        tracker.observe(
            live_event(
                "gesture",
                segment_id=2,
                text="дом",
                final_text="",
                draft_text="я работать дом",
                literal_text="я работать дом",
                sequence=3,
                first_sequence=3,
                last_sequence=3,
                token_count=1,
                status="draft",
            )
        )
        is None
    )

    result = tracker.observe(
        live_event(
            "transcript",
            text="Я работаю.",
            final_text="Я работаю.",
            draft_text="дом",
            literal_text="я работать дом",
            sequence=2,
            first_sequence=1,
            last_sequence=2,
            token_count=2,
            status="enhanced",
            enhanced=True,
        )
    )

    assert result == "Я работаю. дом"


def test_enhanced_tracker_rejects_transcript_before_formatting():
    tracker = EnhancedTranscriptTracker("день")
    tracker.observe(
        live_event(
            "gesture",
            text="день",
            final_text="",
            draft_text="день",
            literal_text="день",
            sequence=1,
            first_sequence=1,
            last_sequence=1,
            token_count=1,
            status="draft",
        )
    )

    with pytest.raises(RuntimeError, match="before formatting"):
        tracker.observe(
            live_event(
                "transcript",
                text="День.",
                final_text="День.",
                draft_text="",
                literal_text="день",
                sequence=1,
                first_sequence=1,
                last_sequence=1,
                token_count=1,
                status="enhanced",
                enhanced=True,
            )
        )


def test_enhanced_tracker_rejects_non_authoritative_full_text():
    tracker = EnhancedTranscriptTracker("день")

    with pytest.raises(RuntimeError, match="non-authoritative full_text"):
        tracker.observe(
            live_event(
                "gesture",
                text="день",
                final_text="",
                draft_text="день",
                full_text="другая строка",
                literal_text="день",
                sequence=1,
                first_sequence=1,
                last_sequence=1,
                token_count=1,
                status="draft",
            )
        )


def test_enhanced_tracker_rejects_non_authoritative_literal_text():
    tracker = EnhancedTranscriptTracker("день")

    with pytest.raises(RuntimeError, match="non-authoritative literal_text"):
        tracker.observe(
            live_event(
                "gesture",
                text="день",
                final_text="",
                draft_text="день",
                literal_text="День был отформатирован.",
                sequence=1,
                first_sequence=1,
                last_sequence=1,
                token_count=1,
                status="draft",
            )
        )


def test_enhanced_tracker_rejects_literal_fallback_for_target_segment():
    tracker = EnhancedTranscriptTracker("день")
    tracker.observe(
        live_event(
            "gesture",
            text="день",
            final_text="",
            draft_text="день",
            literal_text="день",
            sequence=1,
            first_sequence=1,
            last_sequence=1,
            token_count=1,
            status="draft",
        )
    )
    tracker.observe(
        live_event(
            "formatting",
            text="",
            final_text="",
            draft_text="день",
            literal_text="день",
            sequence=1,
            first_sequence=1,
            last_sequence=1,
            token_count=1,
            status="formatting",
        )
    )

    with pytest.raises(RuntimeError, match="was not enhanced"):
        tracker.observe(
            live_event(
                "transcript",
                text="день",
                final_text="день",
                draft_text="",
                literal_text="день",
                sequence=1,
                first_sequence=1,
                last_sequence=1,
                token_count=1,
                status="literal",
                enhanced=False,
            )
        )


def test_legacy_websocket_candidate_still_accepts_raw_gesture():
    assert websocket_candidate({"type": "gesture", "text": "день"}) == "день"


def test_require_enhanced_is_allowed_for_upload_only_mode():
    video = Path(__file__).parent / "data" / "test.mp4"

    args = parse_args(
        [
            "--base-url",
            "https://hack.eferzo.xyz/api",
            "--video",
            str(video),
            "--mode",
            "upload",
            "--require-enhanced",
        ]
    )

    assert args.mode == "upload"
    assert args.require_enhanced


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


def live_event(
    message_type: str,
    *,
    segment_id: int = 1,
    full_text: str | None = None,
    enhanced: bool | None = None,
    **fields,
):
    payload = {
        "type": message_type,
        "segment_id": segment_id,
        **fields,
    }
    payload["full_text"] = (
        full_text
        if full_text is not None
        else " ".join(
            value.strip()
            for value in (payload["final_text"], payload["draft_text"])
            if value.strip()
        )
    )
    if enhanced is not None:
        payload["enhanced"] = enhanced
    return payload
