"""Source separation with Demucs v4 (htdemucs).

The only stage in the pipeline with a real memory ceiling. It recovers from
out-of-memory by halving the segment length, then by moving to CPU.
"""

from __future__ import annotations

import gc
import inspect
import sys

import numpy as np
from numpy.typing import NDArray

from song2midi.device import DeviceBudget
from song2midi.errors import SeparationUnavailableError
from song2midi.separation.base import PassthroughSeparator, Separator

MAX_ATTEMPTS = 3
CPU_FALLBACK_SEGMENT = 4.0


class DemucsSeparator:
    def __init__(self, budget: DeviceBudget, model_name: str = "htdemucs") -> None:
        self.budget = budget
        self.model_name = model_name

    def separate(
        self, audio: NDArray[np.float32], sr: int
    ) -> dict[str, NDArray[np.float32]]:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise SeparationUnavailableError(f"Demucs is not installed: {exc}") from exc

        try:
            model = get_model(self.model_name)
        except Exception as exc:
            raise SeparationUnavailableError(
                f"Could not load the {self.model_name} model: {exc}"
            ) from exc
        model.eval()

        if sr != model.samplerate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=model.samplerate)

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        # Demucs expects the input normalised against its own mono reference.
        reference = wav.mean(dim=1)
        wav = (wav - reference.mean()) / reference.std()

        sources = self._apply_with_retries(apply_model, model, wav, torch)
        sources = sources * reference.std() + reference.mean()

        stems = {
            name: np.ascontiguousarray(
                sources[0, index].cpu().numpy(), dtype=np.float32
            )
            for index, name in enumerate(model.sources)
        }

        del model, sources, wav
        _free_memory(torch)
        return stems

    def _apply_with_retries(self, apply_model, model, wav, torch):
        budget = self.budget
        for attempt in range(MAX_ATTEMPTS):
            try:
                with torch.no_grad():
                    return self._apply(apply_model, model, wav, budget)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise SeparationUnavailableError(f"Demucs failed: {exc}") from exc
                _free_memory(torch)
                if attempt == MAX_ATTEMPTS - 1:
                    break
                if budget.device == "cuda" and attempt >= 1:
                    _warn("separation ran out of GPU memory; falling back to CPU")
                    budget = DeviceBudget("cpu", CPU_FALLBACK_SEGMENT, budget.available_gb)
                else:
                    budget = budget.halved()
                    _warn(
                        f"separation ran out of memory; retrying with "
                        f"segment={budget.segment_seconds}s"
                    )
        raise SeparationUnavailableError("Demucs ran out of memory on every attempt")

    def _apply(self, apply_model, model, wav, budget: DeviceBudget):
        kwargs = {
            "device": budget.device,
            "overlap": 0.25,
            "split": True,
            "progress": False,
        }
        # `segment` moved between apply_model and the model itself across demucs
        # releases; support both rather than pinning a patch version.
        if "segment" in inspect.signature(apply_model).parameters:
            kwargs["segment"] = budget.segment_seconds
        else:
            model.segment = budget.segment_seconds
        return apply_model(model, wav, **kwargs)


def build_separator(separate: bool, budget: DeviceBudget) -> Separator:
    return DemucsSeparator(budget) if separate else PassthroughSeparator()


def _free_memory(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warn(message: str) -> None:
    print(f"song2midi: {message}", file=sys.stderr)
