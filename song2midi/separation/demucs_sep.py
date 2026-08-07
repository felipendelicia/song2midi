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

from song2midi.device import DeviceBudget, is_out_of_memory
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
            except (RuntimeError, MemoryError) as exc:
                if not is_out_of_memory(exc):
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
        # BOTH, and the model attribute is the one that matters.
        #
        # apply_model's `segment` only decides how the mix is sliced. Inside
        # HTDemucs.forward, `use_train_segment` pads every chunk shorter than
        # `model.segment * samplerate` straight back up to it, so slicing alone
        # changes nothing. Measured peak RSS on a 30 s mix, passing segment to
        # apply_model only: 1121 MB at 7.8 s, 1142 MB at 4.0 s, 1171 MB at
        # 2.0 s - flat, and slightly worse for the extra chunks. Setting
        # model.segment as well: 1129 / 844 / 672 MB.
        #
        # Without this the whole memory budget in device.py is inert, and the
        # out-of-memory ladder in _apply_with_retries retries with exactly the
        # memory that just failed.
        _set_segment(model, budget.segment_seconds)
        kwargs = {
            "device": budget.device,
            "overlap": 0.25,
            "split": True,
            # Separation is minutes of silent CPU work — by far the longest
            # stage — and without this the CLI is indistinguishable from hung.
            # demucs writes its bar to stderr, so stdout stays clean for the
            # output path.
            "progress": sys.stderr.isatty(),
        }
        if "segment" in inspect.signature(apply_model).parameters:
            kwargs["segment"] = budget.segment_seconds
        return apply_model(model, wav, **kwargs)


def _set_segment(model, seconds: float) -> None:
    """Set the padding target on the model, not just the slicing width.

    A pretrained htdemucs is a BagOfModels wrapping the real networks, and the
    attribute that drives the padding lives on each inner model.
    """
    for target in getattr(model, "models", [model]):
        if hasattr(target, "segment"):
            target.segment = seconds


def build_separator(separate: bool, budget: DeviceBudget) -> Separator:
    return DemucsSeparator(budget) if separate else PassthroughSeparator()


def _free_memory(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warn(message: str) -> None:
    print(f"song2midi: {message}", file=sys.stderr)
