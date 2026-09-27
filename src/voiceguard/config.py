from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioCfg(BaseConfigModel):
    sr_nb: int = 8000
    sr_wb: int = 16000
    frame_ms: int = 20
    segment_s: float = 1.0


class VadCfg(BaseConfigModel):
    engine: str = "webrtc"
    aggressiveness: int = 2


class LpcCfg(BaseConfigModel):
    order_nb: int = 10
    order_wb: int = 16
    window: str = "hamming"


class D2Cfg(BaseConfigModel):
    windows: list[int] = Field(default_factory=lambda: [8, 16, 32])
    directions: list[str] = Field(default_factory=lambda: ["fwd", "bwd"])
    modes: list[str] = Field(default_factory=lambda: ["bit", "block"])
    train_frac: float = 0.8
    max_train: int = 5000
    mlp_lr: float = 0.001
    seed: int = 42
    alpha: float = 0.01
    chi2_max_w: int = 8


class ChannelsCfg(BaseConfigModel):
    stage0: list[str] = Field(
        default_factory=lambda: [
            "clean",
            "g711a",
            "amrnb_12.2",
            "amrnb_4.75",
            "amrwb_12.65",
            "amrnb_12.2+loss3",
        ]
    )


class FusionCfg(BaseConfigModel):
    sprt_alpha: float = 0.001
    sprt_beta: float = 0.05
    t_max_s: float = 30.0


class EngineCfg(BaseConfigModel):
    stream_d2_checks: str = "all"


class PathsCfg(BaseConfigModel):
    data: str = "data/"
    models: str = "data/models/"
    reports: str = "reports/"


class DebugCfg(BaseConfigModel):
    save_audio: bool = False


class Config(BaseConfigModel):
    audio: AudioCfg = Field(default_factory=AudioCfg)
    vad: VadCfg = Field(default_factory=VadCfg)
    lpc: LpcCfg = Field(default_factory=LpcCfg)
    d2: D2Cfg = Field(default_factory=D2Cfg)
    channels: ChannelsCfg = Field(default_factory=ChannelsCfg)
    fusion: FusionCfg = Field(default_factory=FusionCfg)
    engine: EngineCfg = Field(default_factory=EngineCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    debug: DebugCfg = Field(default_factory=DebugCfg)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML file and apply optional overrides."""
    if path is None:
        path = os.environ.get("VG_CONFIG", "configs/default.yaml")

    config_path = Path(path)
    if not config_path.is_file():
        # Fallback to search relative to project root
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / path
        if candidate.is_file():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {config_path} must be a YAML mapping")

    if overrides:
        data = _deep_update(data, overrides)

    return Config.model_validate(data)
