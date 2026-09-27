from __future__ import annotations

import numpy as np


def _build_alaw_luts() -> tuple[np.ndarray, np.ndarray]:
    pcm16 = np.arange(-32768, 32768, dtype=np.int16)
    pcm = pcm16.astype(np.int32)
    sign = np.where(pcm >= 0, 0x80, 0x00)
    pcm_abs = np.abs(pcm)
    pcm_abs = np.clip(pcm_abs, 0, 32767) >> 3

    seg = np.zeros_like(pcm_abs)
    seg = np.where(pcm_abs >= 32, 1, seg)
    seg = np.where(pcm_abs >= 64, 2, seg)
    seg = np.where(pcm_abs >= 128, 3, seg)
    seg = np.where(pcm_abs >= 256, 4, seg)
    seg = np.where(pcm_abs >= 512, 5, seg)
    seg = np.where(pcm_abs >= 1024, 6, seg)
    seg = np.where(pcm_abs >= 2048, 7, seg)

    mantissa = np.where(seg == 0, (pcm_abs >> 1) & 0x0F, (pcm_abs >> seg) & 0x0F)
    alaw = sign | (seg << 4) | mantissa
    enc_lut = (alaw ^ 0x55).astype(np.uint8)

    alaw_vals = np.arange(256, dtype=np.uint8)
    a = alaw_vals.astype(np.int32) ^ 0x55
    sign_dec = (a & 0x80) != 0
    seg_dec = (a >> 4) & 0x07
    mant_dec = a & 0x0F
    val = np.where(
        seg_dec == 0,
        (mant_dec << 4) + 8,
        ((mant_dec << 4) + 0x108) << (seg_dec - 1),
    )
    val = np.where(sign_dec, val, -val)
    dec_lut = np.clip(val, -32768, 32767).astype(np.int16)
    return enc_lut, dec_lut


def _build_ulaw_luts() -> tuple[np.ndarray, np.ndarray]:
    pcm16 = np.arange(-32768, 32768, dtype=np.int16)
    pcm = pcm16.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0x00)
    mag = np.abs(pcm)
    mag = np.clip(mag + 132, 0, 32767)

    seg = np.zeros_like(mag)
    seg = np.where(mag >= 256, 1, seg)
    seg = np.where(mag >= 512, 2, seg)
    seg = np.where(mag >= 1024, 3, seg)
    seg = np.where(mag >= 2048, 4, seg)
    seg = np.where(mag >= 4096, 5, seg)
    seg = np.where(mag >= 8192, 6, seg)
    seg = np.where(mag >= 16384, 7, seg)

    mant = (mag >> (seg + 3)) & 0x0F
    ulaw = sign | (seg << 4) | mant
    enc_lut = (~ulaw & 0xFF).astype(np.uint8)

    u_vals = np.arange(256, dtype=np.uint8)
    u = ~u_vals.astype(np.int32) & 0xFF
    sign_dec = (u & 0x80) != 0
    seg_dec = (u >> 4) & 0x07
    mant_dec = u & 0x0F
    val = ((mant_dec << 3) + 132) << seg_dec
    val = val - 132
    val = np.where(sign_dec, -val, val)
    dec_lut = np.clip(val, -32768, 32767).astype(np.int16)
    return enc_lut, dec_lut


_ALAW_ENC_LUT, _ALAW_DEC_LUT = _build_alaw_luts()
_ULAW_ENC_LUT, _ULAW_DEC_LUT = _build_ulaw_luts()


def encode_alaw(pcm16: np.ndarray) -> np.ndarray:
    """Encode linear int16 PCM to 8-bit G.711 A-law."""
    idx = pcm16.astype(np.int32) + 32768
    return _ALAW_ENC_LUT[idx]


def decode_alaw(alaw: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 A-law to linear int16 PCM."""
    return np.asarray(_ALAW_DEC_LUT[alaw])


def encode_ulaw(pcm16: np.ndarray) -> np.ndarray:
    """Encode linear int16 PCM to 8-bit G.711 mu-law."""
    idx = pcm16.astype(np.int32) + 32768
    return np.asarray(_ULAW_ENC_LUT[idx])


def decode_ulaw(ulaw: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 mu-law to linear int16 PCM."""
    return np.asarray(_ULAW_DEC_LUT[ulaw])


def transcode_g711a(samples: np.ndarray) -> np.ndarray:
    """Transcode float32 [-1, 1] through G.711 A-law."""
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    encoded = encode_alaw(pcm16)
    decoded = decode_alaw(encoded)
    return np.asarray((decoded.astype(np.float32) / 32767.0).clip(-1.0, 1.0), dtype=np.float32)


def transcode_g711u(samples: np.ndarray) -> np.ndarray:
    """Transcode float32 [-1, 1] through G.711 mu-law."""
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    encoded = encode_ulaw(pcm16)
    decoded = decode_ulaw(encoded)
    return np.asarray((decoded.astype(np.float32) / 32767.0).clip(-1.0, 1.0), dtype=np.float32)
