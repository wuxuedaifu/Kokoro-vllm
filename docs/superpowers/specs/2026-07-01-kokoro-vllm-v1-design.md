# Kokoro TTS on vLLM V1 — Design Spec

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Reference:** [`wuxuedaifu/xttsv2-vllm-streaming-server`](https://github.com/wuxuedaifu/xttsv2-vllm-streaming-server)

## 1. Goal

Serve the Kokoro TTS model **through the vLLM V1 engine** behind a streaming,
OpenAI-compatible HTTP server. The vLLM engine must be genuinely in the
execution path (continuous batching + scheduling of Kokoro forward passes), not
merely emulated at the API layer.

## 2. Background & the core architectural fact

The XTTS-v2 reference is a natural vLLM fit because XTTS-v2 is **autoregressive**:
its GPT backbone generates audio tokens token-by-token, which is exactly what
vLLM's engine (continuous batching, paged KV-cache, token-delta streaming)
accelerates. The reference ports that GPT to vLLM V1 and pipes hidden states out
(via CUDA-IPC) to a HiFi-GAN decoder that lives *outside* vLLM.

**Kokoro is the opposite — fully non-autoregressive.** It is a StyleTTS2 +
iSTFTNet pipeline that runs in a **single forward pass**:

```
CustomAlbert (PL-BERT encoder) → TextEncoder → ProsodyPredictor(ref_s) → iSTFTNet decoder → 24kHz waveform
```

There is no token-by-token generation, no KV-cache, no growing sequence. So
vLLM's decode-loop acceleration does **not** apply. However:

1. **vLLM V1 supports non-autoregressive encoder/pooling models** (BERT-style),
   merged in 2025 (vLLM PR #16188). vLLM's engine can host a single-forward-pass
   model and return per-request tensors through a `Pooler`.
2. The reference's central trick — *"token IDs are placeholders; the real
   semantics live in `cond_latents`," delivered via vLLM's multimodal
   (`SupportsMultiModal`) path* — maps **directly** onto Kokoro, where the voice
   style vector **`ref_s`** is the analogue of `cond_latents`.

This makes a faithful "Kokoro on the vLLM engine" build real, not aspirational.

**What vLLM concretely buys here:** continuous batching + scheduling of every
phoneme-chunk forward pass across all concurrent requests, the request queue, and
worker/GPU lifecycle management. **Honest caveat (accepted):** because Kokoro is
one forward pass, there is no KV-cache benefit — the win is cross-request
batching and serving infrastructure, not decode-loop speedup.

### 2.1 Unavoidable implication — move off ONNX to PyTorch

vLLM requires a PyTorch `nn.Module`. The current repo runs Kokoro via
`kokoro-onnx` (ONNX Runtime), which **cannot** be registered in vLLM. The vLLM
path therefore uses the PyTorch **`kokoro` / `KModel`** package and
`.safetensors` weights. The existing ONNX-based CLI is left untouched.

## 3. Chosen approach — Approach A: whole Kokoro as one vLLM multimodal model

Register the **entire** `KModel` (CustomAlbert → TextEncoder → ProsodyPredictor →
iSTFTNet) as a single vLLM V1 model implementing `SupportsMultiModal` + the
pooling interface.

- Phoneme token IDs = vLLM input tokens (placeholders, as in XTTS).
- Voice vector `ref_s` = multimodal conditioning input (the `cond_latents`
  analogue).
- The model's pooled output = the raw 24 kHz waveform for that chunk (iSTFT runs
  inside the forward).
- Streaming = the frontend splits text into phoneme chunks and submits each as
  its own vLLM request; audio streams out chunk-by-chunk. vLLM continuously
  batches every chunk from every concurrent user.

Because audio is produced *inside* the vLLM worker, it returns via vLLM's
**normal pooling output path** — **no custom CUDA-IPC is required for v1** (unlike
the reference, which needed CUDA-IPC to extract mid-graph hidden states because
its decoder lived outside vLLM). CUDA-IPC remains an optional later optimization
if per-chunk audio-tensor serialization becomes a bottleneck.

### Approaches considered and rejected

- **Approach B — encoder-only on vLLM** (CustomAlbert as a plain pooling model,
  ProsodyPredictor + decoder outside via CUDA-IPC, mirroring the reference's
  topology 1:1). Rejected: CustomAlbert is a tiny fraction of Kokoro's compute;
  the decoder (the bulk) would not be batched by vLLM, requiring a separate
  hand-rolled batcher. Worse throughput, more moving parts, and it barely "uses
  the engine."
- **Approach C — two-stage split** (encoder and decoder each a separate vLLM
  model, chained). Rejected: vLLM isn't designed to pipeline two models in one
  engine; high complexity for no gain over A.

## 4. Architecture & process model

```
┌─ Main process (uvicorn + FastAPI) ──────────────────────────┐
│  /v1/audio/speech                                            │
│    1. Frontend: text ──G2P(misaki/espeak)──► phonemes        │
│    2. Chunker: phonemes → segments (≤510 tokens each)        │
│    3. Voice resolver: name/blend → ref_s[256] per segment    │
│    4. For each segment → AsyncLLM.encode(tokens, mm={ref_s}) │
│    5. Stream returned waveform bytes to client as they land  │
└──────────────────────────┬──────────────────────────────────┘
                           │ vLLM zmq IPC (built-in)
┌──────────────────────────▼──────────────────────────────────┐
│  vLLM V1 EngineCore subprocess (spawned by AsyncLLM)         │
│    KokoroForConditionalGeneration (SupportsMultiModal)       │
│      CustomAlbert → TextEncoder → ProsodyPredictor(ref_s)    │
│        → iSTFTNet decoder → 24kHz waveform (pooled output)   │
│    Continuous batching across ALL chunks from ALL requests   │
└─────────────────────────────────────────────────────────────┘
```

We use vLLM's built-in `AsyncLLM`, which manages the EngineCore subprocess for
us — we do **not** hand-roll the subprocess as the reference did.

## 5. The vLLM model internals

`KokoroForConditionalGeneration` — a vLLM V1 model class implementing
`SupportsMultiModal` + `VllmModelForPooling`, wrapping the PyTorch `KModel`
submodules.

### 5.1 `ref_s` as a multimodal input

vLLM's engine natively carries only token IDs per request. `ref_s` (256-dim) is
per-request side data, so it travels the multimodal channel:

- Register one custom modality `"voice"`, single item = `float16[256]`.
- Request = `{prompt_token_ids: <phoneme ids>, multi_modal_data: {"voice": ref_s}}`,
  with `speed` passed via `mm_processor_kwargs`.
- A `KokoroMultiModalProcessor` validates/normalizes `ref_s` and attaches it to
  the batch. `ref_s` is **not** expanded into placeholder token embeddings — it
  is global conditioning read inside `forward()`. A **single sentinel
  placeholder token** satisfies vLLM's MM plumbing; the real `ref_s` is read
  from the batched `multi_modal_kwargs` in `forward()`.

### 5.2 Forward pass

```
forward(input_ids, positions, ..., **mm_kwargs) -> audio:
    ref_s   = mm_kwargs["voice"]                      # [B, 256] batched by vLLM
    speed   = mm_kwargs.get("speed", 1.0)
    bert_h  = self.bert(input_ids, attn_mask)         # CustomAlbert
    d_en    = self.text_encoder(bert_h)               # TextEncoder
    dur,F0,N,align = self.predictor(d_en, ref_s, speed)  # ProsodyPredictor
    audio   = self.decoder(align, F0, N, ref_s)       # iSTFTNet (+ iSTFT)
    return audio                                       # variable-length 24kHz waveform
```

Everything today handled by the `kokoro-onnx` `create()` call becomes native
`nn.Module`s — one forward, no autoregression.

### 5.3 Waveform as pooled output

`KokoroWaveformPooler` returns the variable-length waveform as the request's
pooling result. vLLM already supports ragged per-request output tensors (how
token-level embeddings of differing lengths return), so `[num_samples]` per
request fits the contract. Main process receives `PoolingRequestOutput.data` =
the chunk's audio samples.

### 5.4 Config & weight loading

- Minimal HF-style `config.json`: `architectures=["KokoroForConditionalGeneration"]`
  + Kokoro hyperparams (hidden dims, vocab, `max_phoneme_len=510`,
  `sample_rate=24000`). Use `--hf-overrides` if the architecture name needs
  forcing.
- Weights load from Kokoro `.safetensors` (convert `kokoro-v1_0.pth` →
  safetensors) via vLLM's `load_weights()` with a name-remap table
  (`KModel` param names → our module tree).
- The **voice pack** (`voices-v1.0.bin`, per-voice `[510,256]` indexed by token
  count) is **not** a model weight — it stays in the main-process frontend
  because `ref_s` selection depends on each chunk's token length. Only the
  resolved `ref_s[256]` crosses into the worker.

### 5.5 Registration

`ModelRegistry.register_model("KokoroForConditionalGeneration", ...)` via a vLLM
plugin entry-point, so `AsyncLLM(model="...", task="embed")` picks it up without
patching vLLM.

## 6. Frontend (main process)

### 6.1 G2P
Text → phonemes via **`misaki`** (Kokoro's official G2P) for `en-us`/`en-gb`,
`espeak-ng` fallback for other languages in the existing voice table (fr, it, ja,
cmn, …). Phonemes → token IDs via Kokoro's phoneme vocab. Preserves current
`--lang` semantics.

### 6.2 Chunking (the streaming unit)
Chunk on **phoneme-token count ≤ 510** (Kokoro's hard limit — today's source of
the `index 510 out of bounds` retry hack), splitting at sentence/clause
boundaries, falling back to word splits for over-long sentences. Because the
frontend now owns G2P, token counts are exact, so the runtime
retry-and-shrink logic in `process_chunk_sequential` is **eliminated**. Each
chunk = one vLLM request = one streamed audio segment.

### 6.3 Voice resolution & blending
Per chunk: `ref_s = voicepack[num_tokens]` (clamped to 509). Blending reuses the
existing `"af_sarah:60,am_adam:40"` grammar — normalized weighted sum of two
voicepacks' vectors — computed in the frontend, so only the final `ref_s[256]`
crosses into the worker. Voice/blend validation messages preserved from the
current `validate_voice`.

### 6.4 Streaming pipeline (per request)
```
phonemes → [chunk1, chunk2, ...]
for each chunk: submit AsyncLLM.encode(tokens, mm={ref_s, speed})  # all in flight, vLLM-batched
stream results IN ORDER → encode to response_format → yield bytes
```
Chunks fan out concurrently (vLLM batches them); bytes are yielded in submission
order so audio is contiguous. **TTFB = first chunk's forward** — inherently low.

## 7. HTTP API (OpenAI-compatible)

- **`POST /v1/audio/speech`** — body `{model, input, voice, response_format,
  speed}`. `voice` accepts single or blend syntax. `stream=true` (or `Accept:
  audio/*` chunked) streams; otherwise returns full buffer. `response_format`:
  `pcm`, `wav`, `mp3`, `opus` (pcm/wav native via `soundfile`; mp3/opus via
  `ffmpeg`).
- **`GET /v1/audio/voices`** — list voices + languages (replaces `--help-voices`).
- **`GET /health`** — liveness (checks engine ready), mirroring the reference.

**Out of scope for this spec:** the EPUB/PDF CLI ingestion. The existing
`kokoro_tts/` CLI stays untouched; wiring it to hit the server is a follow-up,
not part of this build.

## 8. Project structure

New `kokoro_vllm/` package alongside the untouched `kokoro_tts/` CLI:

```
kokoro_vllm/
  model/
    kokoro_vllm_model.py   # KokoroForConditionalGeneration (SupportsMultiModal, pooler)
    modules.py             # CustomAlbert, TextEncoder, ProsodyPredictor, iSTFTNet (torch)
    pooler.py              # KokoroWaveformPooler
    mm_processor.py        # KokoroMultiModalProcessor (ref_s + speed)
    config.py              # HF-style config + weight-name remap
    weights.py             # .pth → safetensors + load_weights remap
    register.py            # ModelRegistry plugin entry-point
  frontend/
    g2p.py                 # misaki/espeak → phoneme token ids
    chunker.py             # ≤510-token sentence/word chunking
    voices.py              # voicepack load, ref_s selection, blending
  server/
    app.py                 # FastAPI: /v1/audio/speech, /v1/audio/voices, /health
    engine.py              # AsyncLLM(task="embed") lifecycle wrapper
    streaming.py           # ordered fan-out + audio encoding (pcm/wav/mp3/opus)
    schemas.py             # pydantic request/response models
  config.py                # server settings (model path, GPU, batch caps)
```

`pyproject.toml` gains an optional extra:
`[project.optional-dependencies] vllm = ["vllm>=…", "torch", "misaki",
"fastapi", "uvicorn"]` — the existing lightweight CLI install is unaffected; the
server is opt-in via `pip install kokoro-tts[vllm]`.

## 9. Error handling

- **G2P / empty input** → 400 with detail; empty chunk skipped.
- **Chunk > 510 after word-split** → hard-truncate + log warning (no silent drop).
- **Unknown voice / blend** → 400 listing valid voices (reuse current validation
  messages).
- **Engine / CUDA OOM or worker death** → 503; `/health` flips unready; AsyncLLM
  surfaces the failed request without killing the server.
- **Client disconnect mid-stream** → cancel that request's outstanding chunks
  (free vLLM slots).
- **`ffmpeg` missing** for mp3/opus → 400 instructing user to use pcm/wav or
  install ffmpeg.

## 10. Testing

- **Unit (no GPU):** chunker boundary / ≤510 cases; voice-blend math vs. current
  `validate_voice`; G2P token-id parity; `mm_processor` `ref_s` shape/validation;
  weight-name remap covers every `KModel` param.
- **Model (1 GPU):** load real weights; single forward `tokens + ref_s →
  waveform`; **A/B parity** — same text+voice through the vLLM path vs. reference
  `kokoro-onnx`/`KModel`, asserting waveform correlation above a threshold
  (guards the port).
- **Server (integration):** `/v1/audio/speech` non-stream returns valid WAV;
  stream yields ordered contiguous audio; concurrent requests batch (throughput >
  serial); bad-voice / empty-input error paths; client-disconnect cancellation.
- **TDD:** each unit lands test-first (test-driven-development skill).

## 11. Open questions / follow-ups (not blocking)

- Exact `vllm` version pin (must include V1 pooling + multimodal support).
- Whether to add CUDA-IPC waveform transport as a later throughput optimization.
- CLI → server integration (deferred follow-up).
- Target GPU / batch-size caps to tune in `kokoro_vllm/config.py`.
