from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voiceguard.config import Config, load_config


def test_default_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    # Check Section 8 requirements
    assert cfg.audio.sr_nb == 8000
    assert cfg.audio.sr_wb == 16000
    assert cfg.audio.frame_ms == 20
    assert cfg.audio.segment_s == 1.0

    assert cfg.vad.engine == "webrtc"
    assert cfg.vad.aggressiveness == 2

    assert cfg.lpc.order_nb == 10
    assert cfg.lpc.order_wb == 16
    assert cfg.lpc.window == "hamming"

    assert cfg.d2.windows == [8, 16, 32]
    assert cfg.d2.directions == ["fwd", "bwd"]
    assert cfg.d2.modes == ["bit", "block"]
    assert cfg.d2.train_frac == 0.8
    assert cfg.d2.max_train == 5000
    assert cfg.d2.mlp_lr == 0.001
    assert cfg.d2.seed == 42
    assert cfg.d2.alpha == 0.01
    assert cfg.d2.chi2_max_w == 8

    assert cfg.channels.stage0 == [
        "clean",
        "g711a",
        "amrnb_12.2",
        "amrnb_4.75",
        "amrwb_12.65",
        "amrnb_12.2+loss3",
    ]

    assert cfg.fusion.sprt_alpha == 0.001
    assert cfg.fusion.sprt_beta == 0.05
    assert cfg.fusion.t_max_s == 30.0

    assert cfg.engine.stream_d2_checks == "reduced"
    assert cfg.paths.data == "data/"
    assert cfg.paths.models == "data/models/"
    assert cfg.paths.reports == "reports/"
    assert cfg.debug.save_audio is False


def test_config_overrides() -> None:
    cfg = load_config(overrides={"d2": {"seed": 7}})
    assert cfg.d2.seed == 7
    # other fields should remain default
    assert cfg.d2.windows == [8, 16, 32]


def test_config_unknown_key_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides={"unknown_section": {"foo": "bar"}})

    with pytest.raises(ValidationError):
        load_config(overrides={"d2": {"unknown_field": 123}})


def test_config_custom_file_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
audio:
  sr_nb: 8000
  sr_wb: 16000
  frame_ms: 20
  segment_s: 1.0
vad:
  engine: silero
  aggressiveness: 3
lpc:
  order_nb: 10
  order_wb: 16
  window: hamming
d2:
  windows: [8, 16]
  directions: [fwd]
  modes: [bit]
  train_frac: 0.8
  max_train: 1000
  mlp_lr: 0.01
  seed: 99
  alpha: 0.05
  chi2_max_w: 8
channels:
  stage0: [clean]
fusion:
  sprt_alpha: 0.01
  sprt_beta: 0.05
  t_max_s: 10
engine:
  stream_d2_checks: reduced
paths:
  data: data/
  models: data/models/
  reports: reports/
debug:
  save_audio: true
""",
        encoding="utf-8",
    )

    cfg = load_config(path=str(custom_yaml))
    assert cfg.vad.engine == "silero"
    assert cfg.d2.seed == 99
    assert cfg.debug.save_audio is True

    # Test via VG_CONFIG env var
    monkeypatch.setenv("VG_CONFIG", str(custom_yaml))
    cfg_env = load_config()
    assert cfg_env.vad.engine == "silero"
    assert cfg_env.d2.seed == 99
