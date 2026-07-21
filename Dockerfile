# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt \
    && python -m pip uninstall --yes pip setuptools wheel

COPY app.py ./app.py

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts \
    && chown app:app /app/artifacts \
    && chmod 0555 /app /app/app.py \
    && chmod 0750 /app/artifacts

USER 10001:10001

EXPOSE 8085

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8085/health', timeout=2).read()"]

# One worker is intentional: every worker would load another 1+ GiB model copy.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085", "--workers", "1", "--limit-concurrency", "4", "--backlog", "16", "--timeout-keep-alive", "5"]
