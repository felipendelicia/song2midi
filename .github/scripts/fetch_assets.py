"""Stage the model weights the frozen build ships.

Only beat_this's checkpoint today: everything else either lives inside a wheel
(basic-pitch's nmp.onnx, torchcrepe's full.pth) or is fetched on demand at
runtime (demucs' htdemucs weights, ~80 MB, and only when separating).

The URL comes from the installed beat_this so it cannot drift from what the
library would have downloaded on its own.

Usage: python .github/scripts/fetch_assets.py [DEST_DIR]
       DEST_DIR defaults to packaging/_assets, which packaging/song2midi.spec
       reads. Re-running is a no-op once the file is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "packaging" / "_assets"
CHECKPOINT_STEM = "final0"
# A sanity floor, not a hash: the point is to notice an HTML error page or a
# truncated transfer, not to pin a version.
MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024


def main(argv: list[str]) -> int:
    dest_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DEST
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{CHECKPOINT_STEM}.ckpt"

    if target.is_file() and target.stat().st_size >= MIN_CHECKPOINT_BYTES:
        print(f"{target} already present ({target.stat().st_size} bytes)")
        return 0

    import torch
    from beat_this.inference import CHECKPOINT_URL

    url = f"{CHECKPOINT_URL}/{CHECKPOINT_STEM}.ckpt"
    partial = target.with_suffix(".ckpt.part")
    print(f"downloading {url} -> {target}")
    torch.hub.download_url_to_file(url, str(partial), progress=True)

    size = partial.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        partial.unlink()
        print(f"downloaded only {size} bytes; refusing to stage it", file=sys.stderr)
        return 1

    # Load it once. A file torch cannot read is worse than no file at all,
    # because the exe would ship it and fail on a user's machine instead of here.
    torch.load(str(partial), map_location="cpu", weights_only=True)
    partial.replace(target)
    print(f"staged {target} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
