# song2midi.spec  --  PyInstaller >= 6.11, Windows x64, one-dir
#
# Build:   pyinstaller --clean --noconfirm song2midi.spec
# Output:  dist/song2midi/song2midi.exe  (+ dist/song2midi/_internal/)

import importlib.util
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
# 1. basic-pitch: only the ONNX graph. The TF SavedModel, the TFLite blob and
#    the CoreML package are dead weight - basic_pitch/__init__.py picks ONNX
#    because TF/coremltools/tflite-runtime are all absent (uv override).
#    ICASSP_2022_MODEL_PATH == Path(basic_pitch.__file__).parent /
#                              "saved_models/icassp_2022/nmp.onnx"
# --------------------------------------------------------------------------
_bp = pkg_dir("basic_pitch")
datas += [(
    str(_bp / "saved_models" / "icassp_2022" / "nmp.onnx"),
    "basic_pitch/saved_models/icassp_2022",
)]

# --------------------------------------------------------------------------
# 2. torchcrepe: torchcrepe/load.py builds
#       os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
#    MonophonicTranscriber._track_crepe passes model="full" -> full.pth only.
#    (Ship tiny.pth too if you ever expose a --crepe-model flag; +2 MB.)
# --------------------------------------------------------------------------
_tc = pkg_dir("torchcrepe")
datas += [(str(_tc / "assets" / "full.pth"), "torchcrepe/assets")]

# --------------------------------------------------------------------------
# 3. demucs: demucs/pretrained.py sets
#       REMOTE_ROOT = Path(__file__).parent / 'remote'
#    and reads REMOTE_ROOT/'files.txt' plus the per-bag <name>.yaml. Only the
#    legacy AWS path needs them (the HuggingFace path is tried first), but
#    that is exactly the fallback you want to keep working. ~2 KB total.
# --------------------------------------------------------------------------
_dm = pkg_dir("demucs")
datas += [(str(p), "demucs/remote") for p in (_dm / "remote").iterdir() if p.is_file()]

# --------------------------------------------------------------------------
# 4. Lazily-imported modules.
#    NOTE: PyInstaller's modulegraph *does* scan nested code objects, so the
#    function-body imports in song2midi (torch, librosa, torchcrepe,
#    demucs.apply, demucs.pretrained, beat_this.inference, basic_pitch.*) are
#    already found. These entries cover the genuinely dynamic cases and act as
#    a tripwire: if one of them disappears the build fails loudly instead of
#    the exe failing at runtime on a user's machine.
# --------------------------------------------------------------------------
hiddenimports += [
    # ONNX is the basic-pitch backend; its absence degrades silently to
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
    # beat_this + its model stack
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
    # resampy 0.4.2 -> pkg_resources.resource_filename (needs setuptools)
    "pkg_resources",
]

# --------------------------------------------------------------------------
# 5. Dead weight. Every one of these is *reachable* by static analysis but
#    never executed by the CLI.
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
    # librosa.display / librosa.util.files are attached lazily and never used
    "matplotlib", "IPython", "notebook",
    # test suites dragged in by collect_submodules
    "numba.tests", "llvmlite.tests", "sklearn.tests", "scipy.tests",
    "numpy.tests", "torch.test", "torch.testing._internal",
    "pytest", "_pytest",
    # NOTE: do NOT exclude `torchgen` - plain `import torch` pulls it in
    # (verified: it lands in sys.modules on a bare `import torch`).
    # Linux/CUDA-only, must never appear in a Windows build
    "triton", "nvidia",
    # GUI toolkits pulled in by transitive optional imports
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
]

# Optional trims, ~70 MB total. Verified on Linux that none of these land in
# sys.modules after importing basic_pitch.inference + demucs.apply +
# demucs.pretrained + torchcrepe + beat_this.inference + librosa + torch.
# They are only pulled in *statically*: sklearn by librosa.segment /
# librosa.decompose (lazy_loader attributes this CLI never touches), sympy /
# networkx / mpmath by torch.fx.experimental. Enable once you have a Windows
# smoke test; leave off if you cannot verify.
# excludes += ["sklearn", "sympy", "networkx", "mpmath"]   # -27, -27, -8, -7 MB

a = Analysis(
    [str(_here / "entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(_here / "rthook_song2midi.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # <- one-dir
    name="song2midi",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # UPX corrupts torch/onnxruntime DLLs
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
