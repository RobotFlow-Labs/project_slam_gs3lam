"""Random Sampling-based Keyframe Mapping helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence


def sample_keyframes(
    keyframes: Sequence,
    *,
    step_idx: int,
    t_opt: int,
    seed: int = 0,
) -> list:
    """Deterministic random keyframe sampling with the current frame always included."""

    if not keyframes:
        return []
    rng = random.Random(seed + step_idx)
    latest = keyframes[-1]
    historical = list(keyframes[:-1])
    if t_opt <= 1 or not historical:
        return [latest]
    sample_count = min(len(historical), t_opt - 1)
    sampled = rng.sample(historical, sample_count)
    sampled.append(latest)
    return sampled
