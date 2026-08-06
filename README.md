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

## Requisitos

Python 3.11 y ffmpeg (solo para mp3/m4a).

Con GPU el pico de VRAM es ~3 GB, en la separación; `device.py` mide la memoria
disponible y elige el `segment` de Demucs que entre con 20% de margen, así que
también funciona en placas de 4 GB. Sin GPU corre en CPU y tarda varios minutos
por canción.

Torch se instala en su variante CPU desde el índice de PyTorch. Para GPU,
instalá el wheel CUDA correspondiente a tu placa por encima.

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
