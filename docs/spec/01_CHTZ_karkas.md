# ЧТЗ-01. Каркас проекта

Основание: общее ТЗ (`docs/spec/00_TZ_obshchee.md`), разделы 5–8, 10. Зависимости: нет.

## 1. Цель

Создать пустой, но рабочий проект: структура каталогов, зависимости, конфигурация, общие типы, логирование, тесты, линтеры, Docker-образ. Все следующие ЧТЗ добавляют код в эту структуру.

## 2. Объём работ

Входит:

- структура репозитория строго по разделу 6 общего ТЗ;
- `pyproject.toml` (Python 3.11, пакет `voiceguard`, src-layout, `uv`);
- `src/voiceguard/types.py` — дословно по разделу 7 общего ТЗ;
- `src/voiceguard/config.py` — загрузка `configs/default.yaml` в pydantic-модели;
- `configs/default.yaml` — по разделу 8 общего ТЗ;
- `src/voiceguard/logging.py` — единая настройка логирования;
- `Makefile`, `pre-commit`, конфигурация `ruff` и `mypy`;
- `docker/Dockerfile.base` — образ с Python и ffmpeg с AMR-энкодерами;
- `README.md` с инструкцией по установке и запуску.

Не входит: любая предметная логика (аудио, модели).

## 3. Функциональные требования

**FR-01.** `voiceguard.config.load_config(path: str | None = None, overrides: dict | None = None) -> Config`. Без аргументов читает `configs/default.yaml`. `overrides` принимает вложенный словарь (`{"d2": {"seed": 7}}`) и применяется поверх файла. Переменная окружения `VG_CONFIG` задаёт путь к альтернативному файлу.

**FR-02.** Модель `Config` (pydantic v2) содержит вложенные модели для каждого раздела YAML: `AudioCfg`, `VadCfg`, `LpcCfg`, `D2Cfg`, `ChannelsCfg`, `FusionCfg`, `EngineCfg`, `PathsCfg`, `DebugCfg` (поле `save_audio: bool = False`). Лишние ключи вызывают ошибку валидации (`extra="forbid"`).

**FR-03.** `voiceguard.logging.get_logger(name: str) -> logging.Logger`. Формат: время ISO-8601, уровень, модуль, сообщение. Уровень задаётся переменной `VG_LOG_LEVEL` (по умолчанию INFO). Отдельно доступен JSON-формат (`VG_LOG_JSON=1`) для движка.

**FR-04.** `Makefile` содержит цели:

| Цель | Действие |
|---|---|
| `install` | `uv sync` со всеми группами зависимостей |
| `test` | `pytest -q` |
| `lint` | `ruff check .` и `mypy src` |
| `fmt` | `ruff format .` |
| `check-ffmpeg` | Скрипт проверки наличия энкодеров `libopencore_amrnb`, `libvo_amrwbenc` и декодеров `amrnb`, `amrwb` |
| `docker-base` | Сборка `docker/Dockerfile.base` |

Цели следующих ЧТЗ (`stage0`, `demo` и др.) добавляются в своих ЧТЗ.

**FR-05.** `docker/Dockerfile.base`: Python 3.11 slim, ffmpeg с AMR-NB/AMR-WB энкодерами (из пакетов дистрибутива, если в них есть нужные энкодеры, иначе сборка ffmpeg с `--enable-version3 --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libvo-amrwbenc`), `uv`, установка проекта. Цель `make check-ffmpeg` внутри образа проходит.

**FR-06.** Группы зависимостей в `pyproject.toml`: основная (numpy, scipy, pandas, pyarrow, scikit-learn, librosa, soundfile, pydantic, pyyaml), `ml` (torch CPU), `speech` (webrtcvad, praat-parselmouth), `data` (datasets, huggingface_hub), `service` (fastapi, uvicorn, httpx, websockets), `ui` (streamlit, plotly), `dev` (pytest, ruff, mypy, pre-commit). Версии закрепляются верхней и нижней границей major-версии.

**FR-07.** `.gitignore` исключает `data/`, `reports/`, модели, `.venv`, кэши.

## 4. Критерии приёмки

- **AC-01.** `make install && make test && make lint` проходят на чистой машине (Linux или macOS).
- **AC-02.** `python -c "from voiceguard.config import load_config; c = load_config(); print(c.d2.windows)"` печатает `[8, 16, 32]`.
- **AC-03.** Тест: `load_config(overrides={"d2": {"seed": 7}}).d2.seed == 7`; неизвестный ключ в YAML вызывает `ValidationError`.
- **AC-04.** `make docker-base` собирает образ; `docker run … make check-ffmpeg` печатает найденные энкодеры и декодеры AMR и завершается с кодом 0.
- **AC-05.** `types.py` совпадает с разделом 7 общего ТЗ; тест создаёт экземпляр каждого типа.

## 5. Результат

Репозиторий с каркасом, README, собранный базовый образ, отчёт о сдаче по пункту 10.7 общего ТЗ.

## 6. Указания агенту

- Не добавлять предметную логику и заглушки модулей других ЧТЗ, кроме пустых `__init__.py`.
- Если в пакетах дистрибутива ffmpeg нет `libvo_amrwbenc`, собрать ffmpeg из исходников в многоэтапной сборке Docker и описать это в README.
