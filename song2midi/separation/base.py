"""The separation boundary."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Separator(Protocol):
    def separate(
        self, audio: NDArray[np.float32], sr: int
    ) -> dict[str, NDArray[np.float32]]: ...


class PassthroughSeparator:
    """No separation.

    The fallback when Demucs is unavailable, and what `--no-separate` selects.
    """

    def separate(
        self, audio: NDArray[np.float32], sr: int
    ) -> dict[str, NDArray[np.float32]]:
        return {"mix": audio}
