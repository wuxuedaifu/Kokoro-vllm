# Kokoro TTS

A CLI text-to-speech tool using the Kokoro model, supporting multiple languages, voices (with blending), and various input formats including EPUB books and PDF documents.

![ngpt-s-c](https://raw.githubusercontent.com/nazdridoy/kokoro-tts/main/previews/kokoro-tts-h.png)

## Features

- Multiple language and voice support
- Voice blending with customizable weights
- EPUB, PDF and TXT file input support
- Standard input (stdin) and `|` piping from other programs
- Streaming audio playback
- Split output into chapters
- Adjustable speech speed
- WAV and MP3 output formats
- Chapter merging capability
- Detailed debug output option
- GPU Support

## Demo

Kokoro TTS is an open-source CLI tool that delivers high-quality text-to-speech right from your terminal. Think of it as your personal voice studio, capable of transforming any text into natural-sounding speech with minimal effort.

https://github.com/user-attachments/assets/8413e640-59e9-490e-861d-49187e967526

[Demo Audio (MP3)](https://github.com/nazdridoy/kokoro-tts/raw/main/previews/demo.mp3) | [Demo Audio (WAV)](https://github.com/nazdridoy/kokoro-tts/raw/main/previews/demo.wav)

## TODO

- [x] Add GPU support
- [x] Add PDF support
- [ ] Add GUI

## Prerequisites

- Python 3.11-3.12 (Python 3.13+ is not currently supported)

## Installation

### Method 1: Install from PyPI (Recommended)

The easiest way to install Kokoro TTS is from PyPI:

```bash
# Using uv (recommended)
uv tool install kokoro-tts

# Using pip
pip install kokoro-tts
```

After installation, you can run:
```bash
kokoro-tts --help
```

### Method 2: Install from Git

Install directly from the repository:

```bash
# Using uv (recommended)
uv tool install git+https://github.com/nazdridoy/kokoro-tts

# Using pip
pip install git+https://github.com/nazdridoy/kokoro-tts
```

### Method 3: Clone and Install Locally

1. Clone the repository:
```bash
git clone https://github.com/nazdridoy/kokoro-tts.git
cd kokoro-tts
```

2. Install the package:

**With `uv` (recommended):**
```bash
uv venv
uv pip install -e .
```

**With `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. Run the tool:
```bash
# If using uv
uv run kokoro-tts --help

# If using pip with activated venv
kokoro-tts --help
```

### Method 4: Run Without Installation

If you prefer to run without installing:

1. Clone the repository:
```bash
git clone https://github.com/nazdridoy/kokoro-tts.git
cd kokoro-tts
```

2. Install dependencies only:

**With `uv`:**
```bash
uv venv
uv sync
```

**With `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run directly:
```bash
# With uv
uv run -m kokoro_tts --help

# With pip (venv activated)
python -m kokoro_tts --help
```

### Download Model Files

After installation, download the required model files to your working directory:

```bash
# Download voice data (bin format is preferred)
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# Download the model
wget https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
```

> The script requires `voices-v1.0.bin` and `kokoro-v1.0.onnx` to be present in the same directory where you run the `kokoro-tts` command.

## Supported voices:

| **Category** | **Voices** | **Language Code** |
| --- | --- | --- |
| 🇺🇸 👩 | af\_alloy, af\_aoede, af\_bella, af\_heart, af\_jessica, af\_kore, af\_nicole, af\_nova, af\_river, af\_sarah, af\_sky | **en-us** |
| 🇺🇸 👨 | am\_adam, am\_echo, am\_eric, am\_fenrir, am\_liam, am\_michael, am\_onyx, am\_puck | **en-us** |
| 🇬🇧 | bf\_alice, bf\_emma, bf\_isabella, bf\_lily, bm\_daniel, bm\_fable, bm\_george, bm\_lewis | **en-gb** |
| 🇫🇷 | ff\_siwis | **fr-fr** |
| 🇮🇹 | if\_sara, im\_nicola | **it** |
| 🇯🇵 | jf\_alpha, jf\_gongitsune, jf\_nezumi, jf\_tebukuro, jm\_kumo | **ja** |
| 🇨🇳 | zf\_xiaobei, zf\_xiaoni, zf\_xiaoxiao, zf\_xiaoyi, zm\_yunjian, zm\_yunxi, zm\_yunxia, zm\_yunyang | **cmn** |

## Usage

### Basic Usage

```bash
kokoro-tts <input_text_file> [<output_audio_file>] [options]
```

> [!NOTE]
> - If you installed via Method 1 (PyPI) or Method 2 (git install), use `kokoro-tts` directly
> - If you installed via Method 3 (local install), use `uv run kokoro-tts` or activate your virtual environment first
> - If you're using Method 4 (no install), use `uv run -m kokoro_tts` or `python -m kokoro_tts` with activated venv

### Commands

- `-h, --help`: Show help message
- `--help-languages`: List supported languages
- `--help-voices`: List available voices
- `--merge-chunks`: Merge existing chunks into chapter files

### Options

- `--stream`: Stream audio instead of saving to file
- `--speed <float>`: Set speech speed (default: 1.0)
- `--lang <str>`: Set language (default: en-us)
- `--voice <str>`: Set voice or blend voices (default: interactive selection)
  - Single voice: Use voice name (e.g., "af_sarah")
  - Blended voices: Use "voice1:weight,voice2:weight" format
- `--split-output <dir>`: Save each chunk as separate file in directory
- `--format <str>`: Audio format: wav or mp3 (default: wav)
- `--debug`: Show detailed debug information during processing

### Input Formats

- `.txt`: Text file input
- `.epub`: EPUB book input (will process chapters)
- `.pdf`: PDF document input (extracts chapters from TOC or content)
- `-` or `/dev/stdin` (Linux/macOS) or `CONIN$` (Windows): Standard input (stdin)

### Examples

```bash
# Basic usage with output file
kokoro-tts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah

# Read from standard input (stdin)
echo "Hello World" | kokoro-tts - --stream
cat input.txt | kokoro-tts - output.wav

# Cross-platform stdin support:
# Linux/macOS: echo "text" | kokoro-tts - --stream
# Windows: echo "text" | kokoro-tts - --stream
# All platforms also support: kokoro-tts /dev/stdin --stream (Linux/macOS) or kokoro-tts CONIN$ --stream (Windows)

# Use voice blending (60-40 mix)
kokoro-tts input.txt output.wav --voice "af_sarah:60,am_adam:40"

# Use equal voice blend (50-50)
kokoro-tts input.txt --stream --voice "am_adam,af_sarah"

# Process EPUB and split into chunks
kokoro-tts input.epub --split-output ./chunks/ --format mp3

# Stream audio directly
kokoro-tts input.txt --stream --speed 0.8

# Merge existing chunks
kokoro-tts --merge-chunks --split-output ./chunks/ --format wav

# Process EPUB with detailed debug output
kokoro-tts input.epub --split-output ./chunks/ --debug

# Process PDF and split into chapters
kokoro-tts input.pdf --split-output ./chunks/ --format mp3

# List available voices
kokoro-tts --help-voices

# List supported languages
kokoro-tts --help-languages
```

> [!TIP]
> If you're using Method 3, replace `kokoro-tts` with `uv run kokoro-tts` in the examples above.
> If you're using Method 4, replace `kokoro-tts` with `uv run -m kokoro_tts` or `python -m kokoro_tts` in the examples above.

## Features in Detail

### EPUB Processing
- Automatically extracts chapters from EPUB files
- Preserves chapter titles and structure
- Creates organized output for each chapter
- Detailed debug output available for troubleshooting

### Audio Processing
- Chunks long text into manageable segments
- Supports streaming for immediate playback
- Voice blending with customizable mix ratios
- Progress indicators for long processes
- Handles interruptions gracefully

### Output Options
- Single file output
- Split output with chapter organization
- Chunk merging capability
- Multiple audio format support

### Debug Mode
- Shows detailed information about file processing
- Displays NCX parsing details for EPUB files
- Lists all found chapters and their metadata
- Helps troubleshoot processing issues

### Input Options
- Text file input (.txt)
- EPUB book input (.epub)
- Standard input (stdin)
- Supports piping from other programs

## vLLM Streaming Server

In addition to the ONNX-based CLI above, this repo ships an **opt-in** HTTP
TTS server (`kokoro_vllm/`) that runs the real PyTorch Kokoro model on top of
[vLLM V1](https://github.com/vllm-project/vllm)'s pooling runner, exposing an
OpenAI-compatible-shaped `/v1/audio/speech` streaming endpoint. It is a
separate code path from the `kokoro-tts` CLI: installing/running it does not
change or replace the existing ONNX CLI in any way.

### 1. Install

The server has its own extra (heavier deps: `vllm`, `torch`, `kokoro`,
`misaki`, `fastapi`, `uvicorn`) so the base CLI install stays light:

```bash
pip install "kokoro-tts[vllm]"
# or, from a checkout:
uv pip install -e ".[vllm]"
```

### 2. Get the weights and convert them

You need the original Kokoro checkpoint (`kokoro-v1_0.pth`) and its
`config.json` (e.g. from the [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
repo on Hugging Face), plus the `voices-v1.0.bin` voicepack archive. Convert
the `.pth` checkpoint into the safetensors + config.json layout vLLM expects:

```bash
python scripts/convert_weights.py \
  --pth kokoro-v1_0.pth \
  --config config.json \
  --out kokoro-model/
```

This writes `kokoro-model/model.safetensors` and `kokoro-model/config.json`
(the latter carries the raw Kokoro config plus the `kmodel_kwargs`/
`architectures`/`model_type` fields vLLM needs to boot the model).

### 3. Run

```bash
python -m kokoro_vllm.server
```

This boots a real vLLM `AsyncLLM` engine (GPU required) and serves on
`0.0.0.0:8000`. Configure it via environment variables (all optional, shown
with their defaults):

| Env var | Default | Meaning |
|---|---|---|
| `KOKORO_MODEL_DIR` | `./kokoro-model` | Directory with `model.safetensors` + `config.json` from step 2 |
| `KOKORO_VOICES_PATH` | `./voices-v1.0.bin` | Voicepack archive |
| `KOKORO_VOCAB_PATH` | `./kokoro-model/config.json` | Phoneme vocab source (the `config.json` the converter wrote in step 2) |
| `KOKORO_DEVICE` | `cuda` | Inference device |
| `KOKORO_SAMPLE_RATE` | `24000` | Output sample rate |

Example:

```bash
KOKORO_MODEL_DIR=./kokoro-model \
KOKORO_VOICES_PATH=./voices-v1.0.bin \
KOKORO_VOCAB_PATH=./kokoro-model/config.json \
python -m kokoro_vllm.server
```

### 4. Call it

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok"}

# List available voices/languages
curl http://localhost:8000/v1/audio/voices

# Synthesize (streamed WAV, written straight to a file)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world.","voice":"af_sarah","response_format":"wav","stream":true}' \
  -o output.wav
```

`voice` also supports blending, e.g. `"af_sarah:70,af_bella:30"`. Set
`"stream": false` for a single buffered response instead of chunk-by-chunk
streaming.

### Honest notes on behavior (read before relying on this in production)

- **Deterministic TTS is a deliberate behavior change.** Upstream Kokoro's
  iSTFTNet decoder injects *unseeded* random noise on every call, so calling
  it twice with identical inputs produces two different waveforms. This
  server seeds the RNG (`KOKORO_SYNTHESIS_SEED`, currently a fixed constant
  in `kokoro_vllm/model/kokoro_vllm_model.py`) immediately before decode, so
  a given `(text, voice, speed)` request yields **bit-identical audio on
  every call**. This is intentional and makes the API cacheable/testable,
  but it is a real, observable difference from vanilla Kokoro — don't expect
  call-to-call variation.

- **What the GPU parity test actually proves.** `tests/model/test_parity_gpu.py`
  shows the vLLM engine's output correlates **1.000000** with the reference
  `kokoro.KModel`. This validates that the port's *plumbing and weight
  conversion* are correct end-to-end (tokenization → multimodal `ref_s` →
  forward pass → pooling → waveform). It is **not** two independent
  implementations agreeing: the vLLM model (`KokoroForConditionalGeneration`)
  runs the *entire* original `KModel` forward pass inside the vLLM worker —
  it wraps Kokoro rather than reimplementing its network. That's by design
  (Approach A: run the whole model on vLLM's pooling runner), and it's why
  correlation is expected to be exactly 1.0 rather than merely "close."

- **Single request at a time is a current, known limitation.** The engine
  runs with `max_num_seqs=1`. Concurrent HTTP requests are accepted and
  processed correctly (streaming and per-request chunk ordering both still
  work) but are **serialized** by the engine — throughput does not scale
  with concurrent clients today. The root cause is that the per-request
  multimodal `ref_s` (the voice embedding) is passed via a `.shared`
  multimodal field, which collapses to a single value when vLLM batches more
  than one request into the same forward pass. Lifting this requires
  upgrading that field from `.shared` to `.batched` multimodal handling — a
  known follow-up, not yet implemented.

- **`mp3`/`opus` output requires `ffmpeg` on `PATH`.** Without it, requests
  for those formats fail fast with a `400`; use `response_format: "pcm"` or
  `"wav"` if `ffmpeg` isn't available.

- **This server is entirely opt-in.** It lives under `kokoro_vllm/`, only
  activates with the `[vllm]` extra installed, and does not touch or affect
  the existing `kokoro-tts` ONNX CLI described above.

## Contributing

This is a personal project. But if you want to contribute, please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)
