from __future__ import annotations

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
    bitrate_kbps: float | None = None  # для AMR: 4.75 … 12.2 (NB), 6.6 … 23.85 (WB)
    loss_pct: float = 0.0  # доля потерянных кадров, %
    snr_db: float | None = None  # None — без шума

    AMR_NB_BITRATES: tuple[float, ...] = (4.75, 5.15, 5.9, 6.7, 7.4, 7.95, 10.2, 12.2)
    AMR_WB_BITRATES: tuple[float, ...] = (
        6.6,
        8.85,
        12.65,
        14.25,
        15.85,
        18.25,
        19.85,
        23.05,
        23.85,
    )
    ALLOWED_CODECS: tuple[str, ...] = ("clean", "g711a", "g711u", "amrnb", "amrwb")

    @classmethod
    def parse(cls, s: str) -> ChannelSpec:
        """Parse channel specification string according to FR-01 grammar."""
        if not s or not isinstance(s, str):
            raise ValueError(f"Channel specification must be a non-empty string, got: {s!r}")

        parts = s.strip().split("+")
        base = parts[0]
        modifiers = parts[1:]

        loss_pct = 0.0
        snr_db: float | None = None

        for mod in modifiers:
            if mod.startswith("loss"):
                try:
                    loss_pct = float(mod[4:])
                except ValueError as err:
                    raise ValueError(f"Invalid loss specification: +{mod}") from err
                if loss_pct < 0.0 or loss_pct > 100.0:
                    raise ValueError(f"Loss percentage must be in [0, 100], got {loss_pct}")
            elif mod.startswith("snr"):
                try:
                    snr_db = float(mod[3:])
                except ValueError as err:
                    raise ValueError(f"Invalid snr specification: +{mod}") from err
            else:
                raise ValueError(f"Unknown channel modifier: +{mod}")

        if "_" in base:
            codec, bitrate_str = base.split("_", 1)
            try:
                bitrate_kbps = float(bitrate_str)
            except ValueError as err:
                raise ValueError(f"Invalid bitrate string: {bitrate_str}") from err
        else:
            codec = base
            bitrate_kbps = None

        if codec not in cls.ALLOWED_CODECS:
            raise ValueError(f"Unknown codec '{codec}'. Allowed codecs: {list(cls.ALLOWED_CODECS)}")

        if codec == "amrnb":
            if bitrate_kbps is None:
                raise ValueError(
                    f"Bitrate is required for amrnb. Allowed bitrates (kbps): "
                    f"{list(cls.AMR_NB_BITRATES)}"
                )
            matched = [b for b in cls.AMR_NB_BITRATES if abs(b - bitrate_kbps) < 1e-4]
            if not matched:
                raise ValueError(
                    f"Invalid bitrate {bitrate_kbps} for amrnb. Allowed bitrates (kbps): "
                    f"{list(cls.AMR_NB_BITRATES)}"
                )
            bitrate_kbps = matched[0]
        elif codec == "amrwb":
            if bitrate_kbps is None:
                raise ValueError(
                    f"Bitrate is required for amrwb. Allowed bitrates (kbps): "
                    f"{list(cls.AMR_WB_BITRATES)}"
                )
            matched = [b for b in cls.AMR_WB_BITRATES if abs(b - bitrate_kbps) < 1e-4]
            if not matched:
                raise ValueError(
                    f"Invalid bitrate {bitrate_kbps} for amrwb. Allowed bitrates (kbps): "
                    f"{list(cls.AMR_WB_BITRATES)}"
                )
            bitrate_kbps = matched[0]
        else:
            if bitrate_kbps is not None:
                raise ValueError(f"Codec '{codec}' does not take a bitrate parameter")

        return cls(
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            loss_pct=loss_pct,
            snr_db=snr_db,
        )

    def __str__(self) -> str:
        parts = [self.codec]
        if self.bitrate_kbps is not None:
            parts.append(f"_{self.bitrate_kbps:g}")
        res = "".join(parts)
        if self.loss_pct > 0.0:
            res += f"+loss{self.loss_pct:g}"
        if self.snr_db is not None:
            res += f"+snr{self.snr_db:g}"
        return res


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
