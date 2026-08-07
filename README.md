# song2midi

Convierte una canción completa a un `.mid` multitrack editable en un DAW.

```bash
uv run song2midi cancion.mp3
```

Genera `cancion.mid` con un track por instrumento: voz, bajo, batería (canal 10,
GM drum map) y el resto.

## Cómo funciona

```
audio → beats → separación (Demucs) → transcripción por stem → cuantización → .mid
```

Cada stem va al transcriptor que mejor lo resuelve:

| stem | transcriptor | por qué |
|------|-------------|---------|
| voz | CREPE | rango amplio, timbre poco armónico |
| bajo | pYIN | no necesita modelo y es preciso en registro grave |
| batería | onsets + bandas espectrales | ver limitaciones |
| resto | Basic Pitch (ONNX) | polifónico genérico |

El tempo sale de `beat_this` (con fallback a librosa) y se guardan los beats
reales, no un BPM constante: cuantizar contra un tempo fijo desalinea el final de
cualquier tema tocado por humanos.

## Opciones

| Flag | Qué hace |
|------|----------|
| `-o, --output` | archivo de salida (default: junto al input) |
| `--device auto\|cpu\|cuda` | dónde correr los modelos |
| `--no-separate` | saltea Demucs y transcribe la mezcla entera |
| `--stems vocals,bass,drums,other` | qué stems transcribir |
| `--quantize 1/16` | alinea los onsets a la grilla |
| `--quantize-strength 0.8` | 0 = sin mover, 1 = snap completo |
| `--workdir DIR` | dónde va el cache |
| `--no-cache` | recomputa todo |

La cuantización está apagada por defecto. El cache está indexado por hash del
archivo, y cada etapa tiene su propia clave: cambiar `--quantize` no invalida los
stems, que son el 90% del tiempo de cómputo.

## Instalación

### Windows

Todavía **no hay `.exe` publicado** — lo construye CI y el workflow no llegó a
correr. Mientras tanto, instalar desde el código fuente funciona igual de bien:

```
git clone https://github.com/felipendelicia/song2midi
cd song2midi
.\scripts\setup-windows.cmd
```

Usá el `.cmd`, no el `.ps1` directamente: Windows viene con la execution policy
en `Restricted` y rechaza cualquier `.ps1`; un `.cmd` no está sujeto a esa
política. El script instala `uv` si falta, baja un Python 3.11 propio sin tocar
el del sistema, arma el entorno y verifica que las dos cosas que suelen romperse
en una máquina nueva funcionen: el runtime de Visual C++ que necesita torch, y
que libsndfile decodifique mp3 sin ffmpeg. Nada de esto pide permisos de
administrador.

Requiere [git](https://git-scm.com/download/win). Si preferís bajar el repo como
ZIP, después de descomprimirlo corré `Get-ChildItem -Recurse -File .\scripts |
Unblock-File`, porque Windows marca los archivos bajados de internet.

Después:

```
uv run song2midi "C:\ruta\cancion.mp3"
```

### Linux / macOS

```bash
git clone https://github.com/felipendelicia/song2midi
cd song2midi && uv sync
uv run song2midi cancion.mp3
```

### Cuánto baja la primera vez

| qué | tamaño | cuándo |
|---|---|---|
| entorno (torch, onnxruntime, numba…) | ~350 MB | `uv sync` |
| checkpoint de `beat_this` | 77 MB | primera detección de tempo |
| pesos de htdemucs | ~80 MB | primera separación |

Todo queda cacheado. Cada canción separada deja además ~170 MB de stems en el
cache del sistema (`%LOCALAPPDATA%\song2midi` en Windows, `~/.cache/song2midi`
en Linux); borralo cuando quieras, se regenera. `--no-separate` no baja los
pesos de htdemucs.

## Requisitos

Python 3.11.

**Formatos:** `.wav`, `.flac`, `.ogg`, `.aiff`, `.mp3` y `.opus` se decodifican
sin herramientas externas — el libsndfile que traen los wheels de `soundfile`
linkea libmpg123 y libopus. Solo `.m4a`, `.aac` y `.wma` necesitan `ffmpeg`, en
el PATH o al lado del ejecutable.

Sin GPU corre en CPU y tarda varios minutos por canción.

## GPU (NVIDIA)

Por defecto se instala torch en variante CPU: el wheel CUDA pesa ~2.4 GB contra
~120 MB, y una máquina sin placa NVIDIA nunca lo usaría. Para activarlo:

```bash
uv sync --locked --no-group cpu --group cu126
```

En Windows, `scripts/setup-windows.cmd` detecta la placa y te lo ofrece.

**Ojo: un `uv sync` a secas revierte a CPU** — es exacto por diseño. Usá
`uv run --no-sync` después de instalar la variante CUDA. Si te olvidás el
`--no-group cpu`, uv falla con un mensaje claro en vez de instalar algo mixto.

Se usa el índice **cu126**, no uno más nuevo, por dos razones verificadas contra
`download.pytorch.org`:

1. Es uno de solo dos que publican torch 2.13.0 **y** torchaudio 2.11.0 para
   Windows/cp311 — el par que ya fija el lockfile.
2. De esos dos, es el único que todavía trae kernels para placas pre-Turing.
   PyTorch compila cu126 para `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` y cu130 de 7.5
   en adelante, porque CUDA 13 eliminó Maxwell, Pascal y Volta. Los wheels son
   SASS puro: una arquitectura faltante no se puede compilar desde PTX, falla en
   el primer kernel. La placa de 4 GB más común es una GTX 1050/1050 Ti, que es
   6.1.

Solo hace falta el driver de NVIDIA; el wheel trae el runtime de CUDA y cuDNN.

### Qué se acelera y qué no

La separación, la voz (CREPE) y la detección de tempo van a la GPU. El bajo usa
pYIN, la batería usa detección de onsets de librosa, y Basic Pitch corre en
onnxruntime — esos tres se quedan en CPU pase lo que pase.

El pico de VRAM del pipeline es **CREPE (~1.3 GB), no la separación (~0.5 GB)**.
Ambas etapas corren de a una y liberan antes de la siguiente, así que los picos
no se suman. Si aun así falta memoria, CREPE baja el batch a la mitad y después
cae a CPU en vez de perder la corrida; la separación parte el `segment`.

## Build del ejecutable de Windows

Lo hace CI (`.github/workflows/build.yml`) en cada push a `main`, y en un tag
`v*` lo adjunta a un Release. Localmente:

```bash
python .github/scripts/fetch_assets.py      # baja el checkpoint de beat_this
uv pip install pyinstaller==6.21.0
pyinstaller packaging/song2midi.spec --clean --noconfirm
```

El bundle es one-dir, no one-file: son cientos de MB y un one-file los
descomprime en `%TEMP%` en **cada** ejecución. El checkpoint de `beat_this`
(77 MB) va adentro, así que la detección de tempo no necesita red; los pesos de
htdemucs (~80 MB) se bajan del hub de Hugging Face la primera vez que separás.

## Limitaciones conocidas

- **La batería es heurística**, no un modelo entrenado: clasifica cada onset por
  dónde está su energía espectral. Confunde toms con kicks y no distingue hi-hat
  abierto de cerrado. Es el punto más débil del pipeline, y es deliberado — no
  hay transcriptor de batería mantenido e instalable por pip que valga la
  dependencia. La interfaz `Transcriber` hace que reemplazarlo sea un cambio
  local.
- El stem `other` sale como una sola pista: dos guitarras no se separan entre sí.
- Es un punto de partida para editar, no una transcripción exacta.

## Desarrollo

```bash
uv run pytest          # rápido, sin modelos
uv run pytest -m ""    # incluye los tests que cargan modelos
```

El núcleo (`Note`, `TempoMap`, `quantize`, `writer`, la segmentación de f0, la
clasificación de batería) son funciones puras que se testean sin cargar un solo
modelo. Los transcriptores se prueban contra audio sintético con tolerancias
explícitas.

Diseño y plan: `docs/superpowers/`.
