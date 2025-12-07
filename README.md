# Sign language inference API

Сервис скачивает модель из S3 по настройкам в `.env`, поднимает FastAPI и принимает запросы от Go-клиента (см. payload из `getRawLiteralFromAPI`).

## Быстрый старт
1. Создай `.env` рядом с `app.py`:
```
S3_BUCKET=your-bucket
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MODEL_KEY=mvit32-2.onnx
CLASS_LIST_KEY=RSL_class_list.txt
MODEL_PATH=artifacts/mvit32-2.onnx
CLASS_LIST_PATH=artifacts/RSL_class_list.txt
DEMO_API_URL=http://localhost:8080/process
USE_MOCK=false
S3_ENDPOINT_URL=<ваш S3 endpoint без /bucket в конце, для R2: https://38ae25cfcc902dddf384e6cbbc6f24e0.r2.cloudflarestorage.com>
FORCE_DOWNLOAD=false
```
Дополнительно: `S3_ENDPOINT_URL` для S3-совместимых стораджей, `NUM_FRAMES` (по умолчанию 32), `INPUT_SIZE` (224).
`USE_MOCK=true` — модель не скачивается/не загружается, ответ всегда `"(Это МОК)"`.
`FORCE_DOWNLOAD=true` — принудительно перекачать модель/классы при старте, даже если файлы уже есть в `artifacts/` (актуально, когда в том же пути лежит старая версия).

2. Установи зависимости (изолировано через venv):
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3. Запусти API:
```
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Docker
```
cp .env.example .env   # заполни переменные
docker compose up --build
```
Compose автоматически подхватывает `.env` в корне; теперь файл не обязателен для запуска, но без S3 переменных модель не скачается.

### Тесты
1. Подготовь файл `tests/data/frame.jpg` (или .png): любой одиночный кадр RGB с рукой/жестом, размер произвольный, но не пустой.
2. (Опционально для видео-теста) Подготовь `tests/data/sample.mp4` с >=32 кадрами; любые кадры/жесты, стандартный h264/mp4.
2. Подними API (локально или через Docker).
3. Запусти `pytest -m integration`.

Тесты шлют: (а) 32 одинаковых кадров из `frame.jpg`; (б) 32 кадров, равномерно выбранных из `sample.mp4`. Оба ждут непустой `text` от `http://localhost:8080/process`.

## Эндпоинты
- `GET /health` → `"OK"`
- `POST /process` принимает JSON:
```
{
  "frames": ["<base64>", "..."],  // [][]byte из Go сериализуется в base64-строки
  "count": 32
}
```
Возвращает `{ "text": "<распознанный жест>" }`.
