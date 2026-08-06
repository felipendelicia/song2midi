<#
.SYNOPSIS
    Set up song2midi on Windows and, optionally, transcribe a song right away.

.DESCRIPTION
    Installs uv if it is missing, downloads a private Python 3.11, creates the
    locked environment, and verifies the pieces most likely to break on a fresh
    Windows box: that the Visual C++ runtime torch needs is present, that
    libsndfile can decode mp3 without ffmpeg, and that the fast test suite
    passes.

    Nothing this script does needs administrator rights and nothing is
    installed system wide: uv goes to $HOME\.local\bin (or wherever
    UV_INSTALL_DIR / XDG_BIN_HOME point) and the environment lives in .venv
    next to this repository. The one prerequisite that does need an
    administrator is the Microsoft Visual C++ redistributable, which the script
    checks for and does not install.

.PARAMETER Song
    Audio file to transcribe once the environment is ready. Optional. May be
    relative to the directory you invoked the script from.

.PARAMETER SkipTests
    Skip the test suite. Faster, but then the first sign of a broken install is
    a failed transcription.

.EXAMPLE
    .\scripts\setup-windows.cmd

.EXAMPLE
    .\scripts\setup-windows.cmd -Song "C:\Users\me\Music\cancion.mp3"

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1

.NOTES
    Windows will not run a .ps1 out of the box. The default execution policy on
    a client SKU is Restricted, so `.\scripts\setup-windows.ps1` fails with
    "running scripts is disabled on this system" before line 1 executes. Use
    the .cmd wrapper next to this file - a .cmd is not subject to the policy -
    or launch this file explicitly as shown above. To unlock the current
    session only:

        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

    If the repository came from a downloaded .zip rather than `git clone`,
    Windows also marks the files as coming from the internet, and even a
    machine already set to RemoteSigned will refuse. `-ExecutionPolicy Bypass`
    skips that zone check; otherwise clear the mark with:

        Get-ChildItem -Recurse -File .\scripts | Unblock-File

    On a machine managed by Group Policy ("Turn on Script Execution" disabled)
    none of this helps: Group Policy outranks every other scope.

    Run this from a terminal. Double-clicking a .ps1 opens it in Notepad, and
    Explorer's "Run with PowerShell" verb destroys the console the moment the
    script ends - the script detects that case and pauses, but a terminal is
    still the better place to read a ten-minute install.
#>

[CmdletBinding()]
param(
    [string] $Song,
    [switch] $SkipTests
)

$ErrorActionPreference = 'Stop'
# Keep the explicit $LASTEXITCODE checks in charge: under PowerShell 7.4+ a
# non-zero native exit code would otherwise throw before we can explain it.
# The variable does not exist under Windows PowerShell 5.1; assigning it there
# is inert.
$PSNativeCommandUseErrorActionPreference = $false

function Write-Step { param([string] $Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Ok { param([string] $Message) Write-Host "    $Message" -ForegroundColor Green }
function Write-Warn { param([string] $Message) Write-Host "    $Message" -ForegroundColor Yellow }

function Format-Elapsed {
    param([int] $Seconds)
    return '{0:d2}:{1:d2}' -f [int]($Seconds / 60), ($Seconds % 60)
}

# One directory listing, one level deep: cheap enough to run every few seconds
# even while uv is writing tens of thousands of files. A recursive size would
# cost more than the work it is reporting on.
function Get-UnpackProgress {
    param([string] $Dir)
    if (-not $Dir -or -not (Test-Path -LiteralPath $Dir)) { return 'downloading' }
    $count = @(Get-ChildItem -LiteralPath $Dir -Directory -ErrorAction SilentlyContinue).Count
    if ($count -eq 0) { return 'downloading' }
    return "$count packages unpacked"
}

# Run a long command while printing a heartbeat.
#
# Without this the script looks hung for minutes: uv draws a progress bar with
# ANSI escapes that the legacy Windows console often will not render, and torch
# alone is a ~200 MB download from download.pytorch.org. Silence uv's bar and
# print something that renders in any console, so "still working" and "wedged"
# are distinguishable.
function Invoke-WithHeartbeat {
    param(
        [Parameter(Mandatory = $true)][string] $Exe,
        [Parameter(Mandatory = $true)][string[]] $ArgumentList,
        [string] $WatchDir,
        [int] $IntervalSeconds = 10
    )

    $previousNoProgress = $env:UV_NO_PROGRESS
    $env:UV_NO_PROGRESS = '1'
    try {
        # -NoNewWindow so the child still writes its own output to this console.
        $proc = Start-Process -FilePath $Exe -ArgumentList $ArgumentList -NoNewWindow -PassThru
        $started = Get-Date
        $nextBeat = $IntervalSeconds
        while (-not $proc.HasExited) {
            Start-Sleep -Milliseconds 500
            $elapsed = [int]((Get-Date) - $started).TotalSeconds
            if ($elapsed -ge $nextBeat) {
                $nextBeat = $elapsed + $IntervalSeconds
                Write-Host "    [$(Format-Elapsed $elapsed)] $(Get-UnpackProgress $WatchDir)" -ForegroundColor DarkGray
            }
        }
        $proc.WaitForExit()
        $total = [int]((Get-Date) - $started).TotalSeconds
        Write-Host "    [$(Format-Elapsed $total)] finished" -ForegroundColor DarkGray
        return $proc.ExitCode
    } finally {
        $env:UV_NO_PROGRESS = $previousNoProgress
    }
}

# Explorer's "Run with PowerShell" verb opens a console *for* this script and
# destroys it the moment the script returns, so every instruction and every
# error would flash past. Detect that launch: explorer.exe started us with the
# script on our command line. An interactive shell has no script there, and a
# shell started from cmd or a terminal has a different parent. Get-Process has
# no .Parent under Windows PowerShell 5.1 - which is exactly what the verb runs
# - so ask CIM, which answers on both editions.
function Test-ExplorerLaunch {
    try {
        $me = Get-CimInstance Win32_Process -Filter "ProcessId = $PID" -ErrorAction Stop
        if ($me.CommandLine -notlike '*setup-windows.ps1*') { return $false }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($me.ParentProcessId)" -ErrorAction Stop
        return $parent.Name -eq 'explorer.exe'
    } catch {
        return $false   # no WMI, recycled parent id: assume a real terminal
    }
}
$script:HoldConsole = Test-ExplorerLaunch

function Wait-ForReader {
    if ($script:HoldConsole) { Read-Host "`nPress Enter to close" | Out-Null }
}

function Fail {
    param([string] $Message)
    Write-Host "`nERROR: $Message" -ForegroundColor Red
    Pop-Location -ErrorAction SilentlyContinue
    Wait-ForReader
    exit 1
}

# Resolve -Song against the caller's directory: past the Push-Location below, a
# relative path would silently resolve against the repository instead. Doing it
# here also means a typo fails in a second rather than after a ~350 MB install.
if ($Song) {
    $resolved = Resolve-Path -LiteralPath $Song -ErrorAction SilentlyContinue
    if (-not $resolved) { Fail "no such file: $Song" }
    # .ProviderPath, not .Path: $Song is handed to a native command later, and
    # .Path would yield a PowerShell drive path that uv cannot open.
    $Song = $resolved.ProviderPath
    if (-not (Test-Path -LiteralPath $Song -PathType Leaf)) { Fail "not a file: $Song" }
}

# Run from the repository root regardless of where the caller invoked this.
# `uv run --no-sync` finds the project by walking up from the current
# directory, so every uv call below depends on this.
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
if (-not (Test-Path -LiteralPath (Join-Path $repo 'pyproject.toml'))) {
    Fail "no pyproject.toml in $repo - is this the song2midi repository?"
}

# torch ships a license file 171 characters below site-packages:
#   torch-<ver>.dist-info/licenses/third_party/kineto/libkineto/third_party/
#   dynolog/third_party/prometheus-cpp/3rdparty/civetweb/src/third_party/
#   duktape-1.8.0/LICENSE.txt
# uv writes it fine (Rust's std passes \\?\ paths to Win32) and nothing at
# runtime opens it, so the install works. But Explorer, `git clean` and Windows
# PowerShell 5.1 all stop at MAX_PATH, so a deep clone leaves a ~1 GB .venv the
# user cannot delete by hand.
# Budget: 259 - 171 - len('\.venv\Lib\site-packages\') = 63. Recompute when the
# torch pin moves.
$longPaths = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
    -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($repo.Length -gt 63 -and $longPaths -ne 1) {
    Write-Step 'Checking the length of this path'
    Write-Warn "this path is $($repo.Length) characters. Past 63, torch's deepest file lands beyond the 260-character MAX_PATH limit."
    Write-Warn 'The install still works, but Explorer, "git clean" and Windows PowerShell will not be able to delete .venv afterwards.'
    Write-Warn 'Fix it either way:'
    Write-Warn '  - clone somewhere shorter, e.g. C:\src\song2midi, or'
    Write-Warn '  - enable long paths once, from an elevated shell:'
    Write-Warn "      Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1 -Type DWord"
    Write-Warn 'If you end up stuck, "uv venv --clear" deletes .venv regardless of its length.'
}

Write-Step 'Checking for uv'
$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($uv) {
    Write-Ok "found: $($uv.Source)"
} else {
    Write-Host '    not found; installing with the official installer (no admin rights needed)'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer chooses its directory from UV_INSTALL_DIR, XDG_BIN_HOME,
    # XDG_DATA_HOME\..\bin or $HOME\.local\bin, and records it in the user's
    # Environment registry key - which never reaches a session that is already
    # running. Re-read PATH from the registry rather than guessing, and keep the
    # same precedence as a fallback for the case where the installer was told
    # not to modify PATH.
    if ($env:UV_INSTALL_DIR) {
        $fallback = $env:UV_INSTALL_DIR
    } elseif ($env:XDG_BIN_HOME) {
        $fallback = $env:XDG_BIN_HOME
    } elseif ($env:XDG_DATA_HOME) {
        $fallback = Join-Path $env:XDG_DATA_HOME '..\bin'
    } else {
        # PowerShell defines $HOME on Windows as $env:USERPROFILE, which is what
        # the installer uses - including on domain machines with a redirected
        # profile, where HOMEDRIVE/HOMEPATH would have differed.
        $fallback = Join-Path $HOME '.local\bin'
    }
    # Append, never prepend: the goal is only to discover the new directory, not
    # to reorder tools the caller deliberately put first in this session.
    $env:PATH = (@(
        $env:PATH
        [Environment]::GetEnvironmentVariable('PATH', 'User')
        [Environment]::GetEnvironmentVariable('PATH', 'Machine')
        $fallback
    ) | Where-Object { $_ }) -join ';'
    $uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $uv) {
        Fail 'uv was installed but is not on PATH in this session - open a new terminal and re-run this script'
    }
    Write-Ok "installed: $($uv.Source)"
}

Write-Step 'Installing Python 3.11'
# pyproject pins >=3.11,<3.12. uv fetches a standalone build from the
# python-build-standalone releases on GitHub; the system Python, whatever
# version it is, is left alone.
Write-Host '    ~30 MB. Usually under a minute.' -ForegroundColor Gray
$code = Invoke-WithHeartbeat -Exe 'uv' -ArgumentList @('python', 'install', '3.11') -IntervalSeconds 10
if ($code -ne 0) {
    Write-Warn 'could not download a managed Python 3.11.'
    Write-Warn 'uv fetches it from github.com/astral-sh/python-build-standalone/releases;'
    Write-Warn 'if GitHub is unreachable, set UV_PYTHON_INSTALL_MIRROR to another source.'
    Write-Warn 'Continuing anyway: uv prefers an existing Python 3.11 over downloading one,'
    Write-Warn 'so the next steps still work if this machine already has 3.11.'
} else {
    Write-Ok 'ready'
}

# torch and onnxruntime link MSVC's C++ runtime but do not ship it: torch\lib
# holds only c10.dll, torch_cpu.dll and friends, every one of which imports
# MSVCP140.dll and VCRUNTIME140_1.dll. Windows does not include the
# redistributable, and the standalone Python uv installs carries only
# vcruntime140*.dll - never msvcp140*. Without it `import torch` dies with
# 'DLL load failed while importing _C'. The mp3 probe further down cannot catch
# this: libsndfile imports VCRUNTIME140.dll but not MSVCP140.dll, so it decodes
# happily on a box where torch cannot load. Check here, before the ~350 MB
# download, and check by loading the DLLs the way torch does rather than by
# looking in System32 - a 32-bit PowerShell host would be redirected to
# SysWOW64 and get the wrong architecture's answer.
$vcRedistMessage = @'
the Microsoft Visual C++ runtime is missing or incomplete. torch and
onnxruntime need it and neither wheel ships it. Install
https://aka.ms/vs/17/release/vc_redist.x64.exe - that installer does ask for
administrator rights - then re-run this script.
'@

Write-Step 'Checking for the Visual C++ runtime'
$crtProbe = @'
import ctypes
import sys

missing = []
for dll in (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_atomic_wait.dll",
):
    try:
        ctypes.CDLL(dll)
    except OSError:
        missing.append(dll)
if missing:
    print("missing: " + ", ".join(missing), file=sys.stderr)
    sys.exit(3)
print("vc runtime ok")
'@
$crtProbe | uv run --no-project --python 3.11 python -
if ($LASTEXITCODE -eq 3) {
    Fail $vcRedistMessage
} elseif ($LASTEXITCODE -ne 0) {
    # uv could not start an interpreter at all: not a runtime problem. The sync
    # below will say so precisely, and the torch import check will re-test the
    # runtime with the interpreter that actually matters.
    Write-Warn 'could not run the probe yet; re-checked after the environment exists'
} else {
    Write-Ok 'present'
}

Write-Step 'Creating the locked environment (~350 MB of wheels; ~1 GB on disk once unpacked)'
Write-Host '    Expect 5-20 minutes depending on the connection. torch alone is a ~200 MB' -ForegroundColor Gray
Write-Host '    download. A heartbeat prints every 10 seconds; as long as it keeps' -ForegroundColor Gray
Write-Host '    printing, the install is alive.' -ForegroundColor Gray
# --locked fails rather than re-resolving, so the environment matches the
# lockfile that CI builds from.
$sitePackages = Join-Path $repo '.venv\Lib\site-packages'
$code = Invoke-WithHeartbeat -Exe 'uv' -ArgumentList @('sync', '--locked') -WatchDir $sitePackages
if ($code -ne 0) {
    Write-Warn '"No interpreter found for Python 3.11" or a download error means there is'
    Write-Warn 'no Python 3.11 on this machine and uv could not fetch one: install Python'
    Write-Warn '3.11 from python.org, or set UV_PYTHON_INSTALL_MIRROR, then re-run.'
    Fail 'uv sync failed. If it mentions building a wheel from source, open an issue with the output - every dependency should have a win_amd64 wheel.'
}
Write-Ok 'installed'

Write-Step 'Verifying that torch and onnxruntime load'
# The definitive version of the runtime check above: the DLLs loading under
# ctypes proves they exist, not that torch's own extension modules resolve.
$torchProbe = @'
import sys

try:
    import torch
    import onnxruntime
except (ImportError, OSError) as exc:
    print(f"import failed: {exc}", file=sys.stderr)
    sys.exit(3)
print(f"torch {torch.__version__}, onnxruntime {onnxruntime.__version__}")
'@
$torchProbe | uv run --no-sync python -
if ($LASTEXITCODE -eq 3) { Fail $vcRedistMessage }
if ($LASTEXITCODE -ne 0) { Fail 'torch or onnxruntime could not be imported - open an issue with the output' }

Write-Step 'Verifying that mp3 decodes without ffmpeg'
# The single most load-bearing assumption of the Windows port: the libsndfile
# bundled in the soundfile wheel links libmpg123, so mp3 needs no external
# decoder. If this fails, every mp3 falls back to an ffmpeg that is probably
# not installed.
$probe = @'
import sys
import soundfile as sf
formats = sf.available_formats()
print(f"libsndfile {sf.__libsndfile_version__}")
missing = [name for name in ("MP3", "FLAC", "OGG", "WAV") if name not in formats]
if missing:
    print("missing formats: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
if "OPUS" not in sf.available_subtypes("OGG"):
    print("no opus support", file=sys.stderr)
    sys.exit(1)
print("mp3, opus, flac, ogg and wav all decode natively")
'@
$probe | uv run --no-sync python -
if ($LASTEXITCODE -ne 0) { Fail 'the installed libsndfile cannot decode mp3' }

Write-Step 'Priming the beat_this checkpoint (77 MB)'
Write-Host '    From a university host that can be slow. A progress bar follows; if it' -ForegroundColor Gray
Write-Host '    stalls, Ctrl-C is safe - the install is already usable without it.' -ForegroundColor Gray
# From source there is no bundled checkpoint: analysis/beats.py finds one only
# under sys._MEIPASS, which PyInstaller sets and uv never does. So the first
# transcription fetches it from a university WebDAV host. Do it here, where a
# proxy or an outage is visible, rather than mid-run where detect() turns the
# failure into a librosa fallback and still exits 0.
$prime = @'
from beat_this.inference import load_checkpoint

from song2midi.analysis.beats import checkpoint

load_checkpoint(checkpoint(), device="cpu")
print("checkpoint ready")
'@
$prime | uv run --no-sync python -
if ($LASTEXITCODE -ne 0) {
    # Warn, never Fail: the librosa fallback is a legitimate degraded mode and
    # must not block an otherwise good install.
    Write-Warn 'could not fetch the beat_this checkpoint from cloud.cp.jku.at.'
    Write-Warn 'The install is fine and transcription still works, but tempo detection'
    Write-Warn 'falls back to librosa: no downbeats, so bar-level quantisation is'
    Write-Warn 'unavailable and the BPM is less reliable. Re-run this script once the'
    Write-Warn 'network or proxy allows it.'
} else {
    Write-Ok "cached under $env:USERPROFILE\.cache\torch\hub\checkpoints"
}

if (-not $SkipTests) {
    Write-Step 'Running the fast test suite'
    uv run --no-sync pytest -q
    if ($LASTEXITCODE -ne 0) { Fail 'the test suite failed - please open an issue with the output' }
    Write-Ok 'all tests passed'
}

Write-Step 'Ready'
Write-Host @'
    Transcribe a song with:

        uv run song2midi "C:\path\to\cancion.mp3"

    The first run that separates downloads the htdemucs weights (~80 MB) from
    the Hugging Face hub. That is the last download left - the beat_this
    checkpoint was fetched above. Both live under %USERPROFILE%\.cache and are
    reused.

    Each separated song also leaves ~170 MB of stems in
    %LOCALAPPDATA%\song2midi\cache. Delete it whenever you like: it regenerates.

    Add --no-separate for a single-track transcription that is much faster. It
    skips Demucs, but not tempo detection, which cannot be turned off.

    If stderr ever says "beat_this unavailable", tempo detection fell back to
    librosa - that is a failed checkpoint fetch, not a broken install, and the
    run still exits 0.
'@ -ForegroundColor Gray

if ($Song) {
    Write-Step "Transcribing $Song"
    Write-Host '    On CPU this takes several minutes per song.' -ForegroundColor Gray
    uv run --no-sync song2midi $Song
    if ($LASTEXITCODE -ne 0) { Fail 'transcription failed' }
}

Pop-Location -ErrorAction SilentlyContinue
Wait-ForReader
