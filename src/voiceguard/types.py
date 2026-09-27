from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Label(str, Enum):
    LIVE = "live"
    SPOOF = "spoof"


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # float32, моно, диапазон [-1, 1]
    sr: int  # 8000 или 16000
    clip_id: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelSpec:
    codec: str  # "clean" | "g711a" | "g711u" | "amrnb" | "amrwb"
    bitrate_kbps: float | None  # для AMR: 4.75 … 12.2 (NB), 6.6 … 23.85 (WB)
    loss_pct: float = 0.0  # доля потерянных кадров, %
    snr_db: float | None = None  # None — без шума
    # строковая форма: "amrnb_12.2+loss3+snr20"; см. ЧТЗ-03


@dataclass(frozen=True)
class Segment:
    samples: np.ndarray  # 1 с речи после VAD
    sr: int
    voiced_mask: np.ndarray  # bool по кадрам 20 мс
    clip_id: str
    index: int  # номер сегмента в клипе


@dataclass(frozen=True)
class D2Result:
    delta: dict[str, float]  # по check_id, напр. "A_bit_w16_fwd"
    z: dict[str, float]  # стандартизованные отклонения от эталона
    s2: float  # max |z|
    chi2_p: dict[str, float]  # p-значения хи-квадрат
    n_symbols: int
    variant: str  # "A" | "B"


@dataclass(frozen=True)
class WindowScore:
    t: float  # время конца сегмента, с
    s1: float | None  # логит D1
    s2: float | None  # статистика D2
    llr: float  # слитое лог-отношение правдоподобия
    Lambda: float  # накопленная статистика SPRT
    verdict: str  # "pending" | "suspicious" | "normal" | "undecided"
