from __future__ import annotations

import numpy as np

from voiceguard.types import ChannelSpec

__all__ = ["ChannelSpec", "random_spec"]


def random_spec(rng: np.random.Generator, pool: list[str]) -> ChannelSpec:
    """Choose and parse a random channel specification from the given pool."""
    if not pool:
        raise ValueError("Channel pool cannot be empty")
    selected_str = str(rng.choice(pool))
    return ChannelSpec.parse(selected_str)
