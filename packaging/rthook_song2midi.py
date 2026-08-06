"""Runtime hook, executed before song2midi is imported.

song2midi/audio/io.py::_load_ffmpeg resolves the decoder with
`shutil.which("ffmpeg")`, which only searches PATH. Putting the bundle
directory on PATH is what makes a co-located ffmpeg.exe discoverable without
touching the transcoding code. Harmless when no ffmpeg.exe is shipped.
"""

import os
import sys

_bundle = getattr(sys, "_MEIPASS", None)
if _bundle:
    os.environ["PATH"] = _bundle + os.pathsep + os.environ.get("PATH", "")
