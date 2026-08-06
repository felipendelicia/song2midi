# song2midi — Diseño

**Fecha:** 2026-08-06
**Estado:** aprobado, pendiente de plan de implementación

## Problema

Convertir una canción completa (audio mezclado, banda completa) a un archivo MIDI multitrack
editable en un DAW. El resultado no busca ser una transcripción perfecta: busca ser un punto
de partida sólido que el usuario corrige a mano.

## Requisitos

- **Entrada:** archivo de audio (mp3, wav, flac, m4a, ogg).
- **Salida:** un `.mid` multitrack, un track por instrumento, batería en canal 10 (GM drum map).
- **Hardware:** CPU por defecto, GPU opcional. Con GPU, el pico de memoria debe entrar en 4 GB
  de VRAM. En CPU debe funcionar con ~3 GB de RAM disponible.
- **Instrumentos:** voz, bajo, batería, y el resto (guitarras/teclados) como una pista.
- **Uso objetivo:** cargar en un DAW y rearreglar.

### Fuera de alcance

- Notación / partitura (el `.mid` se puede importar a MuseScore si hace falta).
- Separación de instrumentos dentro del stem "other" (dos guitarras salen como una pista).
- Detección de letra o fonemas.
- Interfaz gráfica. Es una CLI.

## Arquitectura

Pipeline de etapas, cada una con una interfaz explícita y una implementación intercambiable.
El acoplamiento entre etapas es un tipo de dato, no una clase.

```
audio → [load] → [beats] → [separate] → [transcribe ×N] → [quantize] → [write] → .mid
```

### Estructura de archivos

```
song2midi/
  cli.py                  entrypoint argparse
  config.py               TranscriptionConfig (dataclass, hasheable → cache key)
  pipeline.py             orquesta etapas, resuelve cache
  device.py               resuelve cpu|cuda, calcula presupuesto de memoria
  errors.py               excepciones del dominio
  audio/
    io.py                 load() → waveform + sr; validación de formato
  analysis/
    beats.py              detect() → TempoMap
  separation/
    base.py               Protocol Separator
    demucs_sep.py         htdemucs, chunked
    passthrough.py        devuelve {"mix": audio}
  transcription/
    base.py               Protocol Transcriber
    polyphonic.py         Basic Pitch
    monophonic.py         torchcrepe / pYIN + segmentación de notas
    drums.py              onsets por banda → GM drum map
  midi/
    model.py              Note, Track, TempoMap
    quantize.py           snap a grilla
    writer.py             → .mid con pretty_midi
tests/
docs/superpowers/specs/
```

### Interfaces

Las dos que sostienen el diseño:

```python
class Separator(Protocol):
    def separate(self, audio: NDArray, sr: int) -> dict[str, NDArray]:
        """Devuelve stems nombrados. El passthrough devuelve {"mix": audio}."""

class Transcriber(Protocol):
    def transcribe(self, audio: NDArray, sr: int) -> list[Note]:
        """Notas absolutas en segundos. Sin conocimiento de tempo ni de MIDI."""
```

Todo lo que sigue a la transcripción opera sobre `list[Note]`. Cuantización, tempo y escritura
MIDI se testean sin cargar un solo modelo.

### Modelo de datos

```python
@dataclass(frozen=True)
class Note:
    start: float      # segundos desde el inicio
    end: float
    pitch: int        # MIDI 0-127
    velocity: int     # 1-127

@dataclass(frozen=True)
class Track:
    name: str         # "vocals", "bass", "drums", "other"
    notes: list[Note]
    program: int      # programa General MIDI
    is_drum: bool

@dataclass(frozen=True)
class TempoMap:
    bpm: float
    beats: NDArray        # tiempos de beat en segundos
    downbeats: NDArray    # subconjunto de beats: primer tiempo de cada compás
```

`TempoMap` guarda los beats reales, no solo un BPM: las canciones tocadas por humanos varían
y cuantizar contra un BPM constante desalinea el final del tema.

## Etapas

### 1. Load (`audio/io.py`)

`soundfile` para wav/flac/ogg; `ffmpeg` vía subprocess para mp3/m4a. Salida: `float32` estéreo
a 44100 Hz, que es lo que espera Demucs. Valida al inicio — un formato no soportado falla en el
segundo 0, no después de tres minutos de separación.

### 2. Beats (`analysis/beats.py`)

`beat_this` (modelo chico, PyTorch, corre en CPU) → beats y downbeats. Fallback a
`librosa.beat.beat_track` si `beat_this` no está disponible; librosa da beats decentes y
downbeats pobres, así que en ese caso se marca `downbeats` como vacío y la cuantización a
compás se desactiva.

El BPM se deriva de la mediana de los intervalos entre beats, no del promedio: es robusto a
beats espurios.

### 3. Separación (`separation/demucs_sep.py`)

`htdemucs` (Demucs v4). Cuatro stems: `drums`, `bass`, `other`, `vocals`.

```python
model = get_model("htdemucs")
stems = apply_model(model, audio, segment=budget.segment_seconds,
                    overlap=0.25, split=True, device=budget.device)
```

Cada stem se escribe a disco en el workdir apenas sale y se libera de memoria. El modelo se
descarga (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`) antes de que arranque la
transcripción.

**Presupuesto de memoria.** Demucs es la única etapa pesada:

| segment | VRAM aprox |
|---------|-----------|
| 7.8 s (default) | ~3.0 GB |
| 5 s | ~2.0 GB |
| 4 s | ~1.5 GB |

`device.py` mide la memoria disponible y elige el segment más grande que entre con margen del
20%. Basic Pitch, torchcrepe y la detección de beats están todos por debajo de 500 MB.

### 4. Transcripción

El mapeo stem → transcriptor es una tabla, no un `if`:

| stem | transcriptor | programa GM |
|------|-------------|-------------|
| `vocals` | monophonic (torchcrepe, `fmin=80`, `fmax=1100`) | 53 (Voice Oohs) |
| `bass` | monophonic (pYIN, `fmin=30`, `fmax=400`) | 33 (Electric Bass finger) |
| `drums` | drums (onsets por banda) | — (`is_drum=True`) |
| `other` | polyphonic (Basic Pitch) | 0 (Acoustic Grand Piano) |
| `mix` | polyphonic (Basic Pitch) | 0 |

**Monofónico** (`monophonic.py`). El contorno de f0 se convierte a notas en cuatro pasos:
descartar frames con periodicidad/confianza bajo umbral, convertir Hz → semitonos MIDI
continuos, aplicar filtro de mediana (~5 frames) para matar el jitter de vibrato, y segmentar
cortando donde el pitch redondeado cambia o donde hay un hueco no-vocalizado mayor a 60 ms.
Notas más cortas que 50 ms se descartan. La velocity sale del RMS del segmento, mapeado a 1-127.

Para bajo se usa pYIN en lugar de torchcrepe: no necesita modelo, es preciso en registro grave
y el bajo es la parte más fácil de todo el pipeline.

**Polifónico** (`polyphonic.py`). Basic Pitch vía su backend ONNX (`basic-pitch[onnx]`), que
evita arrastrar TensorFlow entero. `predict()` devuelve `note_events` como tuplas
`(start, end, pitch, amplitude, pitch_bend)`; se descartan los pitch bends y la amplitud se
mapea a velocity.

**Batería** (`drums.py`). Detección de onsets sobre el stem de batería, y clasificación de cada
onset por dónde está su energía:

| clase | criterio | nota GM |
|-------|----------|---------|
| kick | energía dominante < 150 Hz | 36 |
| snare | energía 150–800 Hz + banda ancha | 38 |
| hi-hat | energía dominante > 6 kHz, decaimiento corto | 42 |
| crash | energía > 6 kHz, decaimiento largo | 49 |

**Esto es una heurística, no un modelo entrenado, y es la parte más débil del pipeline.**
Es una decisión consciente: no hay un transcriptor de batería mantenido e instalable por pip
que valga la pena. Confunde toms con kicks y no distingue hi-hat abierto de cerrado. Para el
uso objetivo — editar en un DAW — un patrón de kick/snare/hat aproximado es un punto de partida
utilizable. La ruta de mejora, si molesta, es integrar un modelo tipo ADTOF en una fase
posterior; la interfaz `Transcriber` hace que sea un reemplazo local.

### 5. Cuantización (`midi/quantize.py`)

Opcional, apagada por defecto. Recibe `list[Note]` y un `TempoMap`, y mueve cada onset a la
subdivisión más cercana (1/16 por defecto, configurable). Un parámetro `strength` entre 0 y 1
interpola entre la posición original y la de la grilla, así se puede apretar el timing sin
matar el groove; cuando la cuantización se activa sin especificar strength, el default es 1.0
(snap completo). Los finales de nota se ajustan para preservar la duración, salvo que eso genere
solapamiento en una pista monofónica, en cuyo caso se trunca al siguiente onset.

Función pura. Sin I/O, sin modelos, testeable línea por línea.

### 6. Escritura (`midi/writer.py`)

`pretty_midi`. Un `Instrument` por track, con su programa GM. El track de batería con
`is_drum=True` (pretty_midi lo pone en canal 10 al exportar). El tempo inicial sale del
`TempoMap`.

## Manejo de errores

El principio: una etapa que falla degrada el resultado, no lo aborta.

| falla | comportamiento |
|-------|---------------|
| Demucs no instalado o falla | fallback a passthrough (mix completo → Basic Pitch), warning explícito en stderr |
| CUDA out of memory | reintento con `segment` a la mitad; si vuelve a fallar, cae a CPU con warning |
| `beat_this` no disponible | fallback a librosa; cuantización a compás desactivada |
| stem silencioso o casi vacío | track vacío, sin excepción |
| formato de audio no soportado | error claro en la etapa de load, antes de cargar cualquier modelo |
| archivo corrupto | idem |

Ninguno de estos casos deja un `.mid` a medio escribir: la escritura es el último paso y es
atómica (archivo temporal + rename).

## Cache

Workdir `.song2midi-cache/<sha256(archivo)[:16]>-<hash(config)>/`, con los stems en wav, los
beats en json y las notas por stem en json. Cada etapa chequea si su salida ya existe antes de
correr. Reintentar un tema tras un fallo en la última etapa no recomputa la separación, que es
el 90% del tiempo.

`--no-cache` lo desactiva. El hash de config asegura que cambiar un parámetro invalide lo que
corresponda.

## CLI

```
song2midi INPUT [-o OUT.mid]
          [--device auto|cpu|cuda]
          [--no-separate]              usa el mix entero, sin Demucs
          [--stems vocals,bass,drums,other]
          [--quantize 1/16] [--quantize-strength 0.8]
          [--workdir DIR] [--no-cache]
          [-v]
```

Por defecto: separar, no cuantizar, device auto, salida junto al input con extensión `.mid`.

## Testing

**Núcleo puro** — `Note`, `TempoMap`, `quantize`, `writer`. pytest con fixtures construidas a
mano. Rápido, sin modelos, sin audio. Es donde vive la mayor parte de la lógica que se puede
romper en silencio.

**Transcriptores** — contra audio sintético generado en el propio test: barridos de seno,
secuencias de notas con frecuencias conocidas, ruido con onsets en tiempos conocidos.
Aserciones con tolerancia explícita: pitch exacto, onset ±50 ms, duración ±100 ms.

**Pipeline end-to-end** — un test sobre un clip de 10 segundos, marcado `@pytest.mark.slow`,
excluido de la corrida por defecto. Verifica que sale un `.mid` válido con los tracks esperados.

**Métrica** — `mir_eval.transcription` para F1 de notas contra un ground truth. No es un test
que pase o falle: es un número que se registra, para que "mejoré el transcriptor de voz" sea
una afirmación medible en vez de una impresión.

## Stack

- **Python 3.11.** Basic Pitch 0.4.0 declara soporte hasta 3.11. Se puede intentar 3.12, pero
  3.11 evita la discusión.
- **uv** para el entorno y las dependencias.
- `demucs` 4.1.0, `basic-pitch[onnx]` 0.4.0, `beat_this` 1.1.0, `torchcrepe` 0.0.24,
  `librosa`, `pretty_midi`, `soundfile`, `numpy`.
- `pytest`, `mir_eval` para desarrollo.

Torch entra como dependencia transitiva de demucs. La instalación por defecto es la de CPU;
el soporte GPU es un extra explícito, para no bajar 2 GB de CUDA en una máquina que no la tiene.

## Fases

1. **Baseline end-to-end.** `load` → Basic Pitch sobre el mix → `writer` → `.mid`. Sin
   separación, sin beats, sin cuantización. Un track. El objetivo es tener el esqueleto del
   pipeline, el modelo de datos y los tests del núcleo funcionando.
2. **Separación.** Demucs + `device.py` + cache. Cuatro tracks, todos vía Basic Pitch. Acá
   aparece el manejo de memoria.
3. **Transcriptores especializados.** Monofónico para voz y bajo, heurística de batería. Es
   donde el resultado pasa de "reconocible" a "editable".
4. **Tempo y cuantización.** `beat_this` + `quantize`. Alinea el MIDI a la grilla del DAW.

Cada fase deja el proyecto en un estado usable. Ninguna requiere rediseñar la anterior.
