from __future__ import annotations

import numpy as np
import pytest

from voiceguard.channel.spec import ChannelSpec, random_spec


def test_channel_spec_parse_standard_codecs() -> None:
    # clean
    clean = ChannelSpec.parse("clean")
    assert clean.codec == "clean"
    assert clean.bitrate_kbps is None
    assert clean.loss_pct == 0.0
    assert clean.snr_db is None
    assert str(clean) == "clean"

    # g711a
    g711a = ChannelSpec.parse("g711a")
    assert g711a.codec == "g711a"
    assert g711a.bitrate_kbps is None
    assert str(g711a) == "g711a"

    # g711u
    g711u = ChannelSpec.parse("g711u")
    assert g711u.codec == "g711u"
    assert str(g711u) == "g711u"

    # amrnb
    amr12 = ChannelSpec.parse("amrnb_12.2")
    assert amr12.codec == "amrnb"
    assert amr12.bitrate_kbps == 12.2
    assert str(amr12) == "amrnb_12.2"

    # amrwb
    wb = ChannelSpec.parse("amrwb_12.65")
    assert wb.codec == "amrwb"
    assert wb.bitrate_kbps == 12.65
    assert str(wb) == "amrwb_12.65"


def test_channel_spec_parse_modifiers() -> None:
    spec1 = ChannelSpec.parse("amrnb_4.75+loss3")
    assert spec1.codec == "amrnb"
    assert spec1.bitrate_kbps == 4.75
    assert spec1.loss_pct == 3.0
    assert spec1.snr_db is None
    assert str(spec1) == "amrnb_4.75+loss3"

    spec2 = ChannelSpec.parse("amrwb_12.65+snr20")
    assert spec2.codec == "amrwb"
    assert spec2.bitrate_kbps == 12.65
    assert spec2.loss_pct == 0.0
    assert spec2.snr_db == 20.0
    assert str(spec2) == "amrwb_12.65+snr20"

    spec3 = ChannelSpec.parse("amrnb_12.2+loss3+snr20")
    assert spec3.codec == "amrnb"
    assert spec3.bitrate_kbps == 12.2
    assert spec3.loss_pct == 3.0
    assert spec3.snr_db == 20.0
    assert str(spec3) == "amrnb_12.2+loss3+snr20"

    # Order of modifiers
    spec3_rev = ChannelSpec.parse("amrnb_12.2+snr20+loss3")
    assert spec3_rev.loss_pct == 3.0
    assert spec3_rev.snr_db == 20.0


def test_channel_spec_roundtrip_all_modes() -> None:
    for br in ChannelSpec.AMR_NB_BITRATES:
        s = f"amrnb_{br:g}"
        parsed = ChannelSpec.parse(s)
        assert str(parsed) == s
        assert ChannelSpec.parse(str(parsed)) == parsed

    for br in ChannelSpec.AMR_WB_BITRATES:
        s = f"amrwb_{br:g}"
        parsed = ChannelSpec.parse(s)
        assert str(parsed) == s
        assert ChannelSpec.parse(str(parsed)) == parsed


def test_channel_spec_invalid_bitrates() -> None:
    # Invalid AMR-NB bitrate
    with pytest.raises(ValueError) as excinfo:
        ChannelSpec.parse("amrnb_15.0")
    assert "Invalid bitrate" in str(excinfo.value)
    assert "4.75" in str(excinfo.value)

    # Missing bitrate for AMR-NB
    with pytest.raises(ValueError) as excinfo:
        ChannelSpec.parse("amrnb")
    assert "Bitrate is required" in str(excinfo.value)

    # Invalid AMR-WB bitrate
    with pytest.raises(ValueError) as excinfo:
        ChannelSpec.parse("amrwb_5.0")
    assert "Invalid bitrate" in str(excinfo.value)
    assert "6.6" in str(excinfo.value)

    # Missing bitrate for AMR-WB
    with pytest.raises(ValueError) as excinfo:
        ChannelSpec.parse("amrwb")
    assert "Bitrate is required" in str(excinfo.value)

    # Bitrate on codec that doesn't accept one
    with pytest.raises(ValueError):
        ChannelSpec.parse("g711a_64")

    with pytest.raises(ValueError):
        ChannelSpec.parse("clean_16")


def test_channel_spec_invalid_syntax_and_codecs() -> None:
    with pytest.raises(ValueError):
        ChannelSpec.parse("unknown_codec")

    with pytest.raises(ValueError):
        ChannelSpec.parse("")

    with pytest.raises(ValueError):
        ChannelSpec.parse("amrnb_12.2+unknown3")


def test_random_spec() -> None:
    rng = np.random.default_rng(42)
    pool = ["clean", "g711a", "amrnb_12.2", "amrwb_12.65"]

    chosen = [random_spec(rng, pool) for _ in range(20)]
    assert all(isinstance(c, ChannelSpec) for c in chosen)
    assert all(str(c) in pool for c in chosen)

    # Empty pool raises ValueError
    with pytest.raises(ValueError):
        random_spec(rng, [])
