# Kokoro-vLLM

**Serve the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model through the [vLLM](https://github.com/vllm-project/vllm) V1 engine — behind an OpenAI-compatible, low-latency streaming HTTP server.**

Kokoro is a fast, high-quality 82M-parameter TTS model. This project runs it *inside* vLLM V1 as a custom multimodal **pooling** model, so you get vLLM's serving infrastructure (async engine, request queue, GPU lifecycle) with an OpenAI `/v1/audio/speech` API and chunk-level audio streaming.

> The repository also still ships the original standalone **CLI** (`kokoro-tts`, ONNX-based) for EPUB/PDF/TXT → audio. The vLLM server is an **opt-in** extra and does not affect the CLI.

---

## Highlights

- **OpenAI-compatible API** — `POST /v1/audio/speech`, `GET /v1/audio/voices`, `GET /health`.
- **Streaming** — text is chunked at sentence/≤510-token boundaries and audio streams back chunk-by-chunk.
- **50 voices + blending** — e.g. `"af_sarah:60,am_adam:40"`.
- **Formats** — `pcm`, `wav` (native), `mp3`, `opus` (via `ffmpeg`).
- **Verified correct** — the vLLM-engine output is **bit-for-bit identical** to the reference PyTorch `KModel` (waveform correlation = `1.000000`).
- **Fast** — ~**89 ms** streaming time-to-first-byte and **~48× real-time** synthesis on a single H200.

## Measured performance (single NVIDIA H200)

Input: `"The quick brown fox jumps over the lazy dog."` (~3.7 s of audio).

| Metric | Value |
|---|---:|
| Streaming TTFB | **89 ms** |
| Per-request latency | **80 ms** |
| Synthesis speed | **~48× real-time** |
| Throughput (single GPU) | **~13 req/s** |

Throughput is currently **flat across concurrency** because the engine runs with `max_num_seqs=1` (see [Limitations](#limitations--roadmap)). Scaling levers: run one engine per GPU (≈ ×N GPUs), and the planned batched-`ref_s` upgrade.

---

## How it works (architecture)

Kokoro is **non-autoregressive** (StyleTTS2 + iSTFTNet: text encoder → duration predictor → vocoder, one forward pass), so it doesn't map onto vLLM the way autoregressive LLMs do. Instead of token generation, this project registers the **whole `KModel` forward** as a vLLM V1 model that:

- takes **phoneme token IDs** as the prompt,
- receives the per-request **voice style vector `ref_s`** through vLLM's **multimodal** input channel (the `"voice"` modality), analogous to how conditioning latents are passed to multimodal models,
- runs the full encoder → prosody → iSTFTNet vocoder inside the vLLM worker, and
- returns the **24 kHz waveform** as the model's **pooling output** (task `"plugin"`).

```
HTTP request
  |  text --G2P (misaki/espeak)--> phonemes --> <=510-token chunks
  |  voice --> ref_s[256] (with blending)
  v
FastAPI (async, ordered fan-out)  -->  vLLM V1 AsyncLLM engine
                                         |- KokoroForConditionalGeneration
                                         |  (encoder -> prosody -> iSTFTNet)
                                         <- waveform (pooling output)
  <-- streamed audio (pcm/wav/mp3/opus)
```

No custom CUDA-IPC is needed: the audio is produced inside the worker and returned through vLLM's normal pooling path.

---

## Install

Requires Python 3.11–3.12 and a CUDA GPU for the server.

```bash
# clone, then:
uv pip install -e ".[vllm]"      # or: pip install -e ".[vllm]"
```

The base install (`pip install .`) pulls only the lightweight CLI dependencies; the vLLM server deps live under the `[vllm]` extra.

## Quickstart

**1. Get the model weights** (Kokoro-82M) and convert them to the layout the server loads:

```bash
# download kokoro-v1_0.pth + config.json from hexgrad/Kokoro-82M, and voices-v1.0.bin
python scripts/convert_weights.py \
  --pth kokoro-v1_0.pth --config config.json --out kokoro-model/
# produces kokoro-model/model.safetensors + kokoro-model/config.json
```

**2. Run the server:**

```bash
python -m kokoro_vllm.server
# serves on 0.0.0.0:8000
```

Config via env vars: `KOKORO_MODEL_DIR` (default `./kokoro-model`), `KOKORO_VOICES_PATH` (default `./voices-v1.0.bin`), `KOKORO_VOCAB_PATH` (default `./kokoro-model/config.json`).

**3. Call it:**

```bash
# non-streaming WAV
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Kokoro on vLLM.","voice":"af_sarah","response_format":"wav"}' \
  -o out.wav

# streaming
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Streaming, chunk by chunk.","voice":"af_sarah","response_format":"wav","stream":true}' \
  -o stream.wav

curl -s http://localhost:8000/v1/audio/voices   # list voices + languages
curl -s http://localhost:8000/health
```

### Request fields (`POST /v1/audio/speech`)

| field | default | notes |
|---|---|---|
| `input` | — | text to synthesize (required, non-empty) |
| `voice` | `af_sarah` | single voice, or blend `"v1:60,v2:40"` |
| `response_format` | `wav` | `pcm` \| `wav` \| `mp3` \| `opus` (mp3/opus need `ffmpeg`) |
| `speed` | `1.0` | 0.5–2.0 |
| `lang` | `en-us` | `en-us`, `en-gb`, `fr-fr`, `it`, `ja`, `cmn` |
| `stream` | `false` | chunk-by-chunk streaming |

---

## Notes on behavior (read these)

- **Deterministic output.** Synthesis seeds the RNG (`KOKORO_SYNTHESIS_SEED`, default `0`) before decode, so a given `(text, voice, speed)` yields **bit-identical audio every call**. This is a deliberate change from upstream Kokoro's per-call random iSTFTNet noise, chosen for reproducibility.
- **What the parity result means.** The correlation-`1.0` parity test validates the *port's plumbing and weight conversion* — this project **wraps** Kokoro's network and runs it inside the vLLM worker; it does not reimplement the network. `1.0` means the engine path faithfully reproduces the reference `KModel`, by design.
- **`ffmpeg`** is required for `mp3`/`opus`; otherwise use `pcm`/`wav` (a clean `400` is returned if it's missing).

## Limitations & roadmap

- **Single-request batching (`max_num_seqs=1`) today.** The per-request `ref_s` uses vLLM's `.shared` multimodal field, which would collapse if multiple requests were batched into one forward pass. Concurrent requests are accepted and processed **serially** (streaming/ordering still work; throughput is capped at one stream at a time per engine). The fix — a `.shared`→`.batched` `ref_s` upgrade — is the main path to true continuous batching.
- **Multi-GPU** is not wired: run one server/engine per GPU behind a load balancer for near-linear throughput scaling in the meantime.

---

## The standalone CLI (`kokoro-tts`)

The original ONNX-based CLI is unchanged and remains available for offline EPUB/PDF/TXT → audio, voice blending, chapter splitting, and streaming playback:

```bash
kokoro-tts input.epub --split-output ./chunks/ --format mp3 --voice af_sarah
```

It requires `kokoro-v1.0.onnx` and `voices-v1.0.bin`. See `kokoro-tts --help`.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [`kokoro`](https://github.com/hexgrad/kokoro) / [`misaki`](https://github.com/hexgrad/misaki) by hexgrad.
- [vLLM](https://github.com/vllm-project/vllm).
- The standalone CLI derives from [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts).
