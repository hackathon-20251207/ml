# Sigma Sign — ML Service

**Real-time Russian Sign Language (RSL) recognition, powered by a video-transformer model exported to ONNX.**

This repository is the ML microservice of **Sigma Sign**, a web application that turns Russian Sign Language into text — either live from a webcam or from an uploaded video — for people who are deaf or hard of hearing. It was built during a 48-hour hackathon and is now looking for research collaborators to push the model, dataset, and grammar-level translation further.

🇷🇺 [Читать на русском](README.ru.md)

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18-black)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## Table of contents

- [What this repo does](#what-this-repo-does)
- [Where it sits in the Sigma Sign stack](#where-it-sits-in-the-sigma-sign-stack)
- [The model](#the-model)
- [Repository structure](#repository-structure)
- [Production API (`app.py`)](#production-api-apppy)
  - [API reference](#api-reference)
  - [Configuration](#configuration)
  - [Quick start](#quick-start)
  - [Docker](#docker)
- [Offline / batch inference (`offline_inference/`)](#offline--batch-inference-offline_inference)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Roadmap & open research questions](#roadmap--open-research-questions)
- [Collaborating with us](#collaborating-with-us)
- [Citation](#citation)
- [License](#license)

---

## What this repo does

Given a short clip of someone signing (either streamed frame-by-frame from a browser, or a standalone video file), this service:

1. samples/pads the clip to a fixed number of frames,
2. resizes and normalizes them,
3. runs them through an ONNX-exported video classification model,
4. returns the most likely gesture out of ~1,600 Russian Sign Language classes.

It is intentionally **isolated-gesture recognition**, not continuous sign-language translation with grammar — see [Known limitations](#known-limitations) for why that distinction matters and where we'd like to take it next.

## Where it sits in the Sigma Sign stack

Sigma Sign has three repositories under this organization:

| Repo | Stack | Role |
|---|---|---|
| [`frontend`](https://github.com/HSE-SignLanguage/frontend) | Vue | Captures webcam frames / video upload, displays translated text |
| [`backend`](https://github.com/HSE-SignLanguage/backend) | Go | Session/orchestration layer, forwards frames to this ML service |
| **`ml`** (this repo) | Python / FastAPI | Runs model inference, returns recognized gesture text |

```mermaid
graph LR
    U[User] -->|webcam / video upload| FE[Frontend - Vue]
    FE -->|frames| BE[Backend - Go]
    BE -->|"POST /process"| ML[ML Service - this repo]
    ML -->|downloads on startup| S3[(S3-compatible storage
    e.g. Cloudflare R2)]
    ML -->|"{ text: gesture }"| BE
    BE --> FE
    FE --> U
```

Happy path, end to end:

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant BE as Backend (Go)
    participant ML as ML Service (this repo)
    U->>FE: Grant camera access / upload video
    FE->>BE: Stream frames
    BE->>ML: POST /process {frames: [base64, ...], count}
    ML->>ML: decode -> resize 224x224 -> normalize
    ML->>ML: ONNX Runtime inference (S3D)
    ML-->>BE: {"text": "recognized gesture"}
    BE-->>FE: recognized gesture
    FE-->>U: Show translated text
```

## The model

- **Architecture:** S3D (Separable 3D CNN), exported to ONNX (`s3d.onnx`).
- **Pretraining:** Kinetics-400 (general video/action recognition).
- **Fine-tuning:** [Slovo](https://github.com/hukenovs/slovo) — an open Russian Sign Language dataset — covering the ~1,600 gesture classes listed in [`RSL_class_list.txt`](./RSL_class_list.txt).
- **Why not the Sber baseline ONNX model?** We benchmarked it early on and moved to a fine-tuned S3D checkpoint instead, prioritizing the two things that matter most for a live UX: **inference speed** and **accuracy** on our target vocabulary. Happy to share the comparison notes with anyone digging into the same tradeoff.
- **Inference strategy:** a sliding window of `NUM_FRAMES` (32 by default) frames is fed to the model per prediction; consecutive duplicate predictions and the `no` (no gesture detected) class are collapsed into a clean final sequence.

```mermaid
graph TD
    V["Input: sequence of frames"] --> W1["Window 1: frames 1-32"]
    V --> W2["Window 2: frames 9-40"]
    V --> W3["Window 3: frames 17-48"]
    W1 --> M["S3D model (ONNX Runtime)"]
    W2 --> M
    W3 --> M
    M --> P["Predicted gesture per window"]
    P --> D["Collapse consecutive duplicates + drop 'no'"]
    D --> R["Final gesture sequence"]
```

> **Model performance:** accuracy: 92% and per-window latency on CPU/GPU go here. 
> **Live demo:** now is not publicly reachable unfortunately.

## Repository structure

```
ml/
├── app.py                     # Production FastAPI service (used by the Go backend)
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Tests and local integration tools
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── .env.example
├── RSL_class_list.txt         # id -> gesture label mapping (~1,600 classes)
├── tests/
│   └── data/                  # frame.jpg / sample.mp4 fixtures for integration tests
└── offline_inference/         # Standalone inference, no API/backend required
    ├── model.py                # Predictor class (loads ONNX model directly, no S3 dependency)
    ├── predict_from_video.py   # CLI: run inference on a local video file end-to-end
    └── configs/
        └── config.json         # model path, class list, threshold, topk, clip_len, provider
```

## Production API (`app.py`)

This is the service the Go backend talks to. On startup it downloads the model and class list from S3-compatible storage (we use Cloudflare R2) if they aren't already present locally, loads them into an ONNX Runtime session, and exposes two endpoints.

### API reference

**`GET /health`**
```
200 OK
"OK"
```

**`POST /process`**

Request:
```json
{
  "frames": ["<base64-encoded image bytes>", "..."],
  "count": 32
}
```
`count` must equal `len(frames)`. Frames are individual images (one per video frame), base64-encoded — this is how the Go backend serializes `[][]byte` over JSON.

Response:
```json
{
  "text": "привет",
  "class_id": 1093,
  "confidence": 0.96,
  "candidates": [{"class_id": 1093, "text": "привет", "confidence": 0.96}],
  "accepted": true,
  "reason": null
}
```

For the `no` class, low confidence, or a small top-1/top-2 margin, the service
returns `accepted: false` with an empty `text`. Some transition windows can
still be confident, so the backend additionally confirms a gesture across two
consecutive windows.

Example with `curl` (one static image, repeated — a proper client should send real distinct frames):
```bash
FRAME=$(base64 -i tests/data/frame.jpg)
curl -X POST http://localhost:8085/process \
  -H "Content-Type: application/json" \
  -d "{\"frames\": [$(printf '"%s",' $(yes "$FRAME" | head -32) | sed 's/,$//')], \"count\": 32}"
```

Errors:
- `400` — `count` doesn't match `len(frames)`, or a frame fails to decode.
- `413` — request body exceeds the configured limit.
- `422` — the request does not contain exactly `NUM_FRAMES` frames or violates the schema.
- `503` — the single inference slot is busy; retry later.
- `500` — unexpected internal error (logged server-side).

### Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | — | Bucket to download the model/class list from |
| `AWS_REGION` | — | Region for the S3/R2 client |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Credentials |
| `S3_ENDPOINT_URL` | — | Custom endpoint for S3-compatible storage (e.g. Cloudflare R2) |
| `MODEL_KEY` | `mvit32-2.onnx` | Object key of the model in the bucket |
| `CLASS_LIST_KEY` | `RSL_class_list.txt` | Object key of the class list in the bucket |
| `MODEL_PATH` | `artifacts/mvit32-2.onnx` | Local path the model is downloaded to / loaded from |
| `CLASS_LIST_PATH` | `artifacts/RSL_class_list.txt` | Local path for the class list |
| `NUM_FRAMES` | `32` | Frames per inference window |
| `INPUT_SIZE` | `224` | Frame resize target (square) |
| `USE_MOCK` | `false` | If `true`, skips model loading entirely; `/process` always returns `"(Это МОК)"` — useful for frontend/backend dev without the model |
| `FORCE_DOWNLOAD` | `false` | Re-download artifacts on startup even if already present locally |
| `MODEL_SHA256` / `CLASS_LIST_SHA256` | — | Optional SHA-256 checksums for artifact validation |
| `MIN_CONFIDENCE` / `MIN_MARGIN` | `0.5` / `0.1` | Confidence and top-1/top-2 margin thresholds |
| `TOP_K` | `3` | Number of diagnostic candidates returned |
| `NO_GESTURE_LABELS` / `NO_GESTURE_IDS` | `no` / `14` | Labels and ids representing “no gesture” |
| `MAX_FRAME_BYTES` | `524288` | Maximum decoded frame size |
| `MAX_IMAGE_SIDE` / `MAX_IMAGE_PIXELS` | `2048` / `2000000` | Image dimension limits |
| `MAX_REQUEST_BYTES` | derived | Maximum JSON body size, including chunked requests |
| `INFERENCE_WAIT_SECONDS` | `0.25` | Wait for the single inference slot before `503` |
| `ONNX_THREADS` | `2` | ONNX Runtime CPU thread count |
| `HOST` / `PORT` | `0.0.0.0` / `8085` | Uvicorn bind address |
| `RELOAD` | `false` | Uvicorn autoreload (dev only) |
| `DEMO_API_URL` | — | Used by local demo/testing tooling |


### Quick start

```bash
# 1. Configure
cp .env.example .env   # fill in S3/R2 credentials and MODEL_KEY=s3d.onnx

# 2. Install (isolated venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Run
uvicorn app:app --host 0.0.0.0 --port 8085
```

Prefer to skip the model entirely while working on the frontend/backend? Set `USE_MOCK=true` in `.env` and skip straight to step 3.

### Docker

```bash
cp .env.example .env   # fill in your variables
docker compose up --build
```

`docker-compose.yml` stores the downloaded model and class list in the named
`ml-artifacts` volume so they persist across restarts.
In production, use versioned `MODEL_KEY`/`CLASS_LIST_KEY` values and set both
SHA-256 checksums: replacing an object under the same S3 key cannot otherwise
be detected from the local cache.

## Offline / batch inference (`offline_inference/`)

Sometimes you want to run the model directly against a video file — for evaluation, demos, or debugging — without spinning up the API or the Go backend. That's what this folder is for.

- **`model.py`** — a standalone `Predictor` class. Loads an ONNX model straight from disk (no S3), builds the id→label mapping from a local class list file, and exposes `.predict(frames)` returning top-k labels + confidences (or `None` below `threshold`).
- **`predict_from_video.py`** — reads a video file with OpenCV, resizes frames to 224×224, splits them into sequential (non-overlapping) chunks of `clip_len` frames, runs each chunk through `Predictor`, and prints a de-duplicated gesture sequence.

`configs/config.json` (example):
```json
{
  "model": {
    "path_to_model": "artifacts/s3d.onnx",
    "path_to_class_list": "artifacts/RSL_class_list.txt",
    "provider": "CPUExecutionProvider",
    "threshold": 0.5,
    "topk": 5,
    "clip_len": 32
  }
}
```

Run it:
```bash
cd offline_inference
python predict_from_video.py
```

> **Note:** `VIDEO_PATH` in `predict_from_video.py` is currently a hardcoded absolute path — a good first contribution would be turning it into a CLI argument (`argparse`) so the script is reusable without editing source.

> **Windows note:** on `win32`/`win64`, `model.py` re-encodes the class list from `cp1251` to `utf-8` (a known encoding quirk when the file is read on Windows) and adds OpenVINO execution-provider paths for hardware acceleration. Nothing to configure on Linux/macOS.

**Difference from the production API:** the API (`app.py`) receives pre-decoded frames the Go backend already captured live and uses an overlapping sliding window; `predict_from_video.py` decodes a whole video itself and chunks it sequentially. Same model, two different framing strategies depending on whether you're doing live streaming or one-shot batch analysis.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -m integration
```

Fixtures needed in `tests/data/`:
- `frame.jpg` (or `.png`) — any single RGB frame with a hand/gesture visible.
- *(optional, for the video test)* `sample.mp4` — ≥32 frames, standard H.264/mp4.

Tests send (a) 32 copies of `frame.jpg`, and (b) 32 frames evenly sampled from `sample.mp4`/`test.mp4`, and assert both return a non-empty `text` from a running `http://localhost:8085/process`.

## Known limitations

Being upfront about these — they're exactly the kind of thing we'd love a research collaboration to help solve:

- **Isolated gestures, not continuous signing.** The model recognizes one gesture per window; it doesn't yet model the grammar, non-manual markers (facial expression, mouth patterns), or co-articulation of continuous RSL sentences.
- **Fixed, closed vocabulary.** ~1,600 classes from Slovo — real-world signing (names, neologisms, regional variants) will fall outside this set.
- **No confidence calibration across window boundaries** — overlapping windows can each fire independently; there's no temporal smoothing/voting beyond simple de-duplication.
- **Single signer framing assumptions** — frames are resized to a fixed square without hand/pose-based cropping, so signer distance/position relative to the camera affects accuracy.
- **Benchmark numbers not yet published** here (see TODO above) — happy to share on request while we finalize an evaluation protocol.

## Roadmap & open research questions

- Continuous sign language recognition (sentence-level, not isolated gesture-level).
- Incorporating non-manual markers (facial expression, mouth shape) which carry grammatical meaning in RSL.
- Expanding vocabulary beyond Slovo's ~1,600 classes with additional data collection.
- Temporal smoothing / voting across overlapping windows instead of simple de-duplication.
- On-device / mobile export (quantization, smaller backbone) for lower-latency inference.
- Formal accuracy/latency benchmarking protocol and public leaderboard.

## Collaborating with us

Sigma Sign started as a hackathon project (December 2025) built to make everyday communication more accessible for the deaf and hard-of-hearing community. We're now looking to partner with researchers working on sign language recognition, continuous gesture translation, or accessibility-focused ML.

If any of the open questions above overlap with your research — reach out. *(contact: email: kuznetsova4ka@gmail.com)*
