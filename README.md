# VoiceGuard AM

VoiceGuard AM — тестовый образец (MVP) системы обнаружения синтезированной (дипфейк) армянской речи в телефонном канале.

## Требования

- **ОС:** Linux или macOS
- **Python:** 3.11 (управляется через `uv`)
- **uv:** >= 0.4.0 ([руководство по установке uv](https://docs.astral.sh/uv/))
- **Docker:** для сборки базового контейнера и запуска телефонного стенда
- **FFmpeg:** с поддержкой энкодеров `libopencore_amrnb`, `libvo_amrwbenc` и декодеров `amrnb`, `amrwb`

> **Примечание по FFmpeg:**
> В стандартных пакетах дистрибутива Debian (включая `python:3.11-slim`) пакет ffmpeg собирается без библиотеки `libvo_amrwbenc` из-за лицензионных ограничений. В базовом Docker-образе (`docker/Dockerfile.base`) используется многоэтапная сборка (multi-stage build), где FFmpeg компилируется из исходников с флагами `--enable-version3 --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libvo-amrwbenc`.

## Запуск демо (MVP)

### Быстрый старт

```bash
# 1. Установить зависимости
make install

# 2. Загрузить данные (Hy-Generated, 300+300 клипов, ~20-40 мин)
make data

# 3. Обучить D1, построить эталоны D2, сформировать отчёт этапа 0
make stage0
# → Отчёт: reports/stage0/report.md
# → Модели: data/models/d1_nb.joblib, data/models/d2_ref_*.json

# 4. Запустить сервер
make demo
# → Откройте http://localhost:8000
```

> **Примечание по каналу amrnb_12.2:** требует Docker с образом `voiceguard-base`
> (собирается командой `make docker-base`). По умолчанию в UI используется канал `clean`.

### Эндпоинты API

| Endpoint | Метод | Описание |
|---|---|---|
| `/` | GET | Веб-интерфейс |
| `/v1/analyze` | POST | Загрузка файла (wav/mp3/m4a/ogg/webm, до 60 с) + параметр `channel` |
| `/v1/stream` | WS | Потоковый PCM16 16 кГц; каждую секунду речи → JSON WindowScore |
| `/v1/passport` | GET | Паспорт детектора (JSON) |
| `/v1/health` | GET | Проверка состояния |



## Конфигурация

Параметры по умолчанию задаются в `configs/default.yaml`.
В коде конфигурация загружается функцией `voiceguard.config.load_config(path, overrides)`:

```python
from voiceguard.config import load_config

# Загрузка стандартной конфигурации
config = load_config()

# Переопределение параметров в рантайме
custom_config = load_config(overrides={"d2": {"seed": 7}})
```

Путь к конфигурационному файлу можно также переопределить через переменную окружения `VG_CONFIG`.

## Логирование

Единая точка логирования — `voiceguard.logging.get_logger`:

```python
from voiceguard.logging import get_logger

logger = get_logger("my_module")
logger.info("Ready")
```

Поддерживаются переменные окружения:
- `VG_LOG_LEVEL`: уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`, по умолчанию `INFO`).
- `VG_LOG_JSON=1`: вывод логов в формате JSON.
