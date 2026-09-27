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

## Установка и запуск

### 1. Локальная установка

Для автоматической синхронизации виртуального окружения и всех групп зависимостей (`dev`, `ml`, `speech`, `data`, `service`, `ui`):

```bash
make install
```

Команда выполняет `uv sync --all-groups`, создавая `.venv` с Python 3.11 и всеми необходимыми пакетами.

### 2. Запуск тестов

```bash
make test
```

### 3. Запуск линтеров

```bash
make lint
```

Проверяет код с помощью `ruff check .` и `mypy src`.

### 4. Форматирование кода

```bash
make fmt
```

### 5. Проверка FFmpeg

Для проверки наличия необходимых кодеков AMR (NB/WB) в вашей системе:

```bash
make check-ffmpeg
```

### 6. Сборка базового Docker-образа

Сборка образа с Python 3.11, скомпилированным FFmpeg (с AMR-NB и AMR-WB энкодерами/декодерами) и всеми зависимостями:

```bash
make docker-base
```

Проверка кодеков внутри собранного образа:

```bash
docker run --rm voiceguard-base:latest make check-ffmpeg
```

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
