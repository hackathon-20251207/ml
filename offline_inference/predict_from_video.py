import cv2
import numpy as np
import json
import os
from offline_inference.model import Predictor  # ← Используем их готовый класс

# === 1. Загрузка конфига ===
with open("configs/config.json", "r") as f:
    config = json.load(f)

# === 2. Инициализация модели ===
# Важно: передаем ПОЛНЫЙ конфиг, а не только путь к модели
predictor = Predictor(config["model"])

# === 3. Укажи путь к видео ===
VIDEO_PATH = "/Users/macbookvera/Desktop/ИИ/HSE SignLanguage/pleasework/tests/video.mp4"  # ← замени на свой файл

# Проверяем, существует ли файл
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

# === 4. Извлечение и обработка кадров ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Не удалось открыть видео: {VIDEO_PATH}")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Меняем размер под модель (224x224) и конвертируем в RGB
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    frames.append(rgb)
cap.release()

print(f"Загружено {len(frames)} кадров.")

# === 5. Подготовка чанков для предсказания ===
clip_len = config["model"]["clip_len"]
clips = []

for i in range(0, len(frames), clip_len):
    clip = frames[i : i + clip_len]
    if len(clip) == clip_len:
        clips.append(clip)

if not clips:
    raise ValueError("Видео слишком короткое: нет полного чанка кадров.")

print(f"Подготовлено {len(clips)} чанков по {clip_len} кадров.")

# === 6. Предсказание ===
print("\nПредсказания:")
all_gestures = []

for i, clip in enumerate(clips):
    # Конвертируем в numpy массив
    clip_array = np.array(clip)

    # Делаем предсказание
    result = predictor.predict(clip_array)

    if result is None:
        gesture = "no"
    else:
        # Берем самый вероятный жест (top-1)
        gesture = (
            result["labels"][0]
            if isinstance(result["labels"], dict)
            else result["labels"][0]
        )

    all_gestures.append(gesture)
    print(f"  Чанк {i + 1}: {gesture}")

# === 7. Итог (уникальные жесты без повторов) ===
final_sequence = []
for g in all_gestures:
    if g not in ["no", ""] and (not final_sequence or g != final_sequence[-1]):
        final_sequence.append(g)

print("\nИтоговая последовательность жестов:")
if final_sequence:
    print(" → ".join(final_sequence))
else:
    print("Не распознано жестов")
