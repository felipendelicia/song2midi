# song2midi.spec  --  PyInstaller >= 6.11, Windows x64, one-dir
#
# Build:   python .github/scripts/fetch_assets.py
#          pyinstaller packaging/song2midi.spec --clean --noconfirm
# Output:  dist/song2midi/song2midi.exe  (+ dist/song2midi/_internal/)
#
# Run it from the repository root: PyInstaller does not chdir to the spec's
# directory, so the paths built from SPECPATH below are what keep the location
# of the spec irrelevant while the working directory stays the repo root.

import importlib.util
import os
import pathlib


_here = pathlib.Path(SPECPATH)


def pkg_dir(name: str) -> pathlib.Path:
    """Directory of an installed package, resolved from the build venv."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"{name} is not importable in the build environment")
    return pathlib.Path(spec.origin).parent


datas = []
binaries = []
hiddenimports = []

# --------------------------------------------------------------------------
# 0. Build-time capability check.
#    song2midi/audio/io.py decodes mp3 and opus through libsndfile rather than
#    ffmpeg. That only holds if the libsndfile this environment links was built
#    with libmpg123 and libopus - true of the soundfile wheels for Windows and
#    macOS, not guaranteed of a distro build. Freezing an exe whose libsndfile
#    cannot read an mp3 would silently drop the format most users feed it, so
#    fail here rather than ship it.
# --------------------------------------------------------------------------
import soundfile as _sf

if "MP3" not in _sf.available_formats():
    raise RuntimeError(
        "the libsndfile in this build environment cannot decode MP3 "
        f"(libsndfile {_sf.__libsndfile_version__}); the frozen exe would "
        "reject .mp3. Install soundfile from a wheel, not from an sdist."
    )
if "OPUS" not in _sf.available_subtypes("OGG"):
    raise RuntimeError("the libsndfile in this build environment has no Opus support")

# --------------------------------------------------------------------------
# 1. basic-pitch: only the ONNX graph (230 KB). The TF SavedModel, the TFLite
#    blob and the CoreML package are dead weight - basic_pitch/__init__.py
#    picks ONNX because TF, coremltools and tflite-runtime are all absent
#    (see the uv override in pyproject.toml).
#    ICASSP_2022_MODEL_PATH == Path(basic_pitch.__file__).parent /
#                              "saved_models/icassp_2022/nmp.onnx"
# --------------------------------------------------------------------------
_bp = pkg_dir("basic_pitch")
datas += [
    (
        str(_bp / "saved_models" / "icassp_2022" / "nmp.onnx"),
        "basic_pitch/saved_models/icassp_2022",
    )
]

# --------------------------------------------------------------------------
# 2. torchcrepe (89 MB): torchcrepe/load.py builds
#       os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
#    MonophonicTranscriber._track_crepe passes model="full", so only full.pth
#    is reachable. Ship tiny.pth too if a --crepe-model flag ever appears.
# --------------------------------------------------------------------------
_tc = pkg_dir("torchcrepe")
datas += [(str(_tc / "assets" / "full.pth"), "torchcrepe/assets")]

# --------------------------------------------------------------------------
# 3. demucs: demucs/pretrained.py sets REMOTE_ROOT = Path(__file__).parent /
#    'remote' and reads files.txt plus the per-bag <name>.yaml. Only the legacy
#    AWS path needs them - the HuggingFace path is tried first - but that is
#    exactly the fallback worth keeping alive. ~2 KB total.
#
#    The htdemucs weights themselves are NOT bundled: ~80 MB fetched from the
#    HF hub on first separation. That is a defensible follow-up, and a separate
#    decision from beat_this, because a user can turn separation off with
#    --no-separate but cannot turn tempo detection off.
# --------------------------------------------------------------------------
_dm = pkg_dir("demucs")
datas += [(str(p), "demucs/remote") for p in (_dm / "remote").iterdir() if p.is_file()]

# --------------------------------------------------------------------------
# 4. beat_this checkpoint (77 MB), staged by .github/scripts/fetch_assets.py.
#    beat_this.inference.load_checkpoint() tries torch.load on the path it is
#    given before it reaches for the network, and song2midi.analysis.beats
#    .checkpoint() hands it this file when frozen. So tempo detection needs no
#    network and cannot quietly fall back to librosa on an offline machine.
#    The filename matters: PyInstaller keeps the basename and beats.py looks
#    for beat_this/final0.ckpt.
# --------------------------------------------------------------------------
_ckpt = _here / "_assets" / "final0.ckpt"
if not _ckpt.is_file():
    raise RuntimeError(
        f"beat_this checkpoint missing at {_ckpt}. "
        "Run `python .github/scripts/fetch_assets.py` before building."
    )
datas += [(str(_ckpt), "beat_this")]  # -> _internal/beat_this/final0.ckpt

# --------------------------------------------------------------------------
# 5. Optional ffmpeg. Not a dependency and not downloaded here: wav, flac, ogg,
#    aiff, mp3 and opus all decode through libsndfile, and only .m4a/.aac/.wma
#    need an external binary. If SONG2MIDI_FFMPEG points at one it is bundled,
#    and song2midi.audio.io.ffmpeg_executable() finds it under sys._MEIPASS.
#
#    Bundle only an LGPL-configured build. The common prebuilt Windows ffmpeg
#    binaries - including the one imageio-ffmpeg ships and the one inside the
#    `av` wheels - are built --enable-gpl --enable-libx264, which would impose
#    GPL on the whole redistributed exe.
# --------------------------------------------------------------------------
_ffmpeg = os.environ.get("SONG2MIDI_FFMPEG")
if _ffmpeg:
    _ffmpeg_path = pathlib.Path(_ffmpeg)
    if not _ffmpeg_path.is_file():
        raise RuntimeError(f"SONG2MIDI_FFMPEG={_ffmpeg} is not a file")
    binaries += [(str(_ffmpeg_path), ".")]

# --------------------------------------------------------------------------
# 6. Lazily-imported modules.
#    PyInstaller's modulegraph does scan nested code objects, so the
#    function-body imports in song2midi (torch, librosa, torchcrepe,
#    demucs.apply, demucs.pretrained, beat_this.inference, basic_pitch.*) are
#    already found. These entries cover the genuinely dynamic cases, and they
#    double as a tripwire: if one disappears the build fails loudly instead of
#    the exe failing at runtime on a user's machine.
# --------------------------------------------------------------------------
hiddenimports += [
    # ONNX is the basic-pitch backend. Its absence degrades silently to
    # "File ... cannot be loaded into either TensorFlow, CoreML, TFLite or ONNX".
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    # demucs resolves htdemucs through the HF hub first; hf_xet is imported
    # from inside huggingface_hub functions and is the default transfer path.
    "huggingface_hub",
    "hf_xet",
    # demucs/hf.py: getattr(importlib.import_module(module), name) where
    # `module` comes from the safetensors metadata -> "demucs.htdemucs".
    # Also the target of the legacy torch.hub unpickle.
    "demucs.htdemucs",
    "demucs.hdemucs",
    "demucs.demucs",
    "demucs.states",
    "demucs.hf",
    "demucs.repo",
    # beat_this and its model stack
    "beat_this.inference",
    "beat_this.model.beat_tracker",
    "beat_this.model.postprocessor",
    "beat_this.model.roformer",
    "rotary_embedding_torch",
    "soxr",
    # torchcrepe pulls these at import time
    "torchcrepe",
    "torchaudio",
    "resampy",
    # resampy 0.4.2 -> pkg_resources.resource_filename, which needs setuptools.
    # This is why pyproject pins setuptools<81: 81 removed pkg_resources.
    "pkg_resources",
]

# --------------------------------------------------------------------------
# 7. Dead weight. Every one of these is reachable by static analysis but never
#    executed by the CLI.
# --------------------------------------------------------------------------
excludes = [
    # basic-pitch probes these at import time and we deliberately do not ship them
    "tensorflow", "tensorflow.lite", "tflite_runtime", "coremltools", "keras",
    # demucs training stack (demucs.pretrained.get_model_from_sig imports it)
    "demucs.train", "demucs.solver", "demucs.evaluate", "demucs.augment",
    "demucs.distrib", "demucs.wav", "demucs.grids", "demucs.separate",
    "dora", "hydra", "omegaconf", "musdb", "museval", "submitit", "treetable",
    "diffq",
    # basic-pitch training stack
    "basic_pitch.train", "basic_pitch.data", "basic_pitch.visualize",
    "basic_pitch.models", "basic_pitch.nn", "basic_pitch.layers",
    # librosa.display and librosa.util.files are attached lazily and never used
    "matplotlib", "IPython", "notebook",
    # test suites dragged in by collect_submodules
    "numba.tests", "llvmlite.tests", "sklearn.tests", "scipy.tests",
    "numpy.tests", "torch.test", "torch.testing._internal",
    "pytest", "_pytest",
    # NOTE: do NOT exclude `torchgen` - a plain `import torch` pulls it in.
    # Linux/CUDA-only, must never appear in a Windows build
    "triton", "nvidia",
    # GUI toolkits pulled in by transitive optional imports
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
]

# Optional trims, ~60 MB of source. None of these land in sys.modules after
# importing basic_pitch.inference + demucs.apply + demucs.pretrained +
# torchcrepe + beat_this.inference + librosa + torch; they are only reachable
# statically (sklearn via librosa.segment/decompose, which this CLI never
# touches; sympy/networkx/mpmath via torch.fx.experimental). Turn them on only
# after the Windows smoke test is green without them, then re-run it.
# excludes += ["sklearn", "sympy", "networkx", "mpmath"]

a = Analysis(
    [str(_here / "entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # No runtime hooks. An earlier draft prepended sys._MEIPASS to PATH so that
    # shutil.which could find a co-located ffmpeg; audio/io.py now looks in the
    # bundle and next to the executable directly, which is the same fix without
    # mutating the environment of every child process.
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # one-dir
    name="song2midi",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX corrupts torch and onnxruntime DLLs
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="song2midi",
)
