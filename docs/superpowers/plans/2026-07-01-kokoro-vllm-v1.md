# Kokoro-on-vLLM-V1 Streaming Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the Kokoro TTS model through the vLLM V1 engine as a `SupportsMultiModal` pooling model (whole `KModel` forward in-worker, `ref_s` as multimodal conditioning, waveform as pooled output), behind an OpenAI-compatible streaming FastAPI server.

> NOTE: the code as merged is the source of truth. Several code snippets in this plan (notably the Task 5 chunker and the Task 9–13 vLLM model/pooler/engine examples) were written before the vLLM 0.24.0 / kokoro APIs were verified; the implementation corrected them (see docs/superpowers/plans/vllm-interface-notes.md and the task commits). Read the code for current behavior.

**Architecture:** A new `kokoro_vllm/` package. The frontend (main process) does G2P → phoneme tokens → ≤510-token chunks → per-chunk `ref_s[256]`, then submits each chunk as a vLLM `AsyncLLM.encode()` request carrying `ref_s` via the multimodal channel. The registered `KokoroForConditionalGeneration` model runs the entire `KModel` forward inside the vLLM worker and returns the 24 kHz waveform as its pooled output. The FastAPI layer fans chunks out concurrently (vLLM batches them) and streams audio bytes back in submission order.

**Tech Stack:** Python 3.11–3.12, PyTorch, `vllm` (V1, pinned), `kokoro` (`KModel`), `misaki` + `espeak-ng` (G2P), FastAPI + uvicorn, `soundfile`/`numpy`, `ffmpeg` (mp3/opus), pytest.

## Global Constraints

- Python requirement: `>=3.11, <3.13` (from `pyproject.toml`) — verbatim.
- Do **not** modify the existing `kokoro_tts/` CLI package; the ONNX CLI must keep working unchanged.
- The vLLM server is an **opt-in** install: all new heavy deps live under the `[project.optional-dependencies] vllm` extra. The base `kokoro-tts` install must remain unchanged in weight.
- Kokoro hard limit: **≤510 phoneme tokens** per forward (padded to 512 with leading/trailing `0`). Never submit a chunk exceeding this.
- Sample rate is fixed at **24000 Hz**; audio dtype is float32 in `[-1, 1]`.
- `ref_s` is a `[1, 256]` (batched `[B, 256]`) tensor: `ref_s[:, :128]` → decoder, `ref_s[:, 128:]` → prosody predictor. Selected as `voicepack[num_tokens]` where `num_tokens = len(input_ids)` before padding, clamped to `[0, 509]`.
- API surface is OpenAI-compatible: `POST /v1/audio/speech`, `GET /v1/audio/voices`, `GET /health`.
- vLLM's pooling/multimodal method signatures are version-sensitive. Every task that touches a vLLM base class includes a step to verify the signature against the **installed** vLLM before implementing.
- TDD throughout: failing test → run-fails → minimal impl → run-passes → commit.

---

## File Structure

```
kokoro_vllm/
  __init__.py
  config.py                # ServerSettings (model path, voices path, gpu, batch caps, vllm version pin note)
  model/
    __init__.py
    kmodel_access.py       # load PyTorch KModel + submodules, expose forward_with_tokens
    hf_config.py           # HF-style config dict + KokoroConfig
    weights.py             # .pth -> safetensors conversion + param name remap table
    mm_processor.py        # KokoroMultiModalProcessor (ref_s + speed)
    pooler.py              # KokoroWaveformPooler
    kokoro_vllm_model.py   # KokoroForConditionalGeneration (SupportsMultiModal + VllmModelForPooling)
    register.py            # ModelRegistry.register_model plugin entry-point
  frontend/
    __init__.py
    vocab.py               # phoneme vocab load + phonemes->input_ids (with [0,...,0] padding)
    g2p.py                 # text -> phonemes (misaki en, espeak fallback)
    chunker.py             # phonemes -> [<=510-token chunks]
    voices.py              # voicepack load, ref_s selection, blend parsing/math
  server/
    __init__.py
    engine.py              # AsyncLLM(task="embed") lifecycle wrapper
    schemas.py             # pydantic request/response models
    streaming.py           # ordered fan-out + audio encoding (pcm/wav/mp3/opus)
    app.py                 # FastAPI app + routes + error handling
tests/
  frontend/
    test_vocab.py
    test_g2p.py
    test_chunker.py
    test_voices.py
  model/
    test_weights_remap.py
    test_mm_processor.py
    test_pooler.py
    test_parity_gpu.py     # marked @pytest.mark.gpu
  server/
    test_schemas.py
    test_streaming.py
    test_app.py
scripts/
  convert_weights.py       # CLI wrapper around model/weights.py
```

---

## Task 1: Package scaffold + `vllm` optional extra

**Files:**
- Create: `kokoro_vllm/__init__.py`, `kokoro_vllm/config.py`
- Create: `kokoro_vllm/model/__init__.py`, `kokoro_vllm/frontend/__init__.py`, `kokoro_vllm/server/__init__.py`
- Modify: `pyproject.toml` (add optional-dependencies + pytest markers)
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: `kokoro_vllm.config.ServerSettings` dataclass with fields
  `model_dir: str = "./kokoro-model"`, `voices_path: str = "./voices-v1.0.bin"`,
  `vocab_path: str = "./config.json"`, `device: str = "cuda"`,
  `max_num_seqs: int = 64`, `sample_rate: int = 24000`,
  `max_phoneme_tokens: int = 510`, and classmethod `from_env() -> ServerSettings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
import os
from kokoro_vllm.config import ServerSettings

def test_defaults():
    s = ServerSettings()
    assert s.sample_rate == 24000
    assert s.max_phoneme_tokens == 510
    assert s.device == "cuda"

def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("KOKORO_DEVICE", "cpu")
    monkeypatch.setenv("KOKORO_MAX_NUM_SEQS", "8")
    s = ServerSettings.from_env()
    assert s.device == "cpu"
    assert s.max_num_seqs == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kokoro_vllm'`

- [ ] **Step 3: Create the package files**

```python
# kokoro_vllm/__init__.py
"""Kokoro TTS served through the vLLM V1 engine."""
```

```python
# kokoro_vllm/config.py
import os
from dataclasses import dataclass


@dataclass
class ServerSettings:
    model_dir: str = "./kokoro-model"
    voices_path: str = "./voices-v1.0.bin"
    vocab_path: str = "./config.json"
    device: str = "cuda"
    max_num_seqs: int = 64
    sample_rate: int = 24000
    max_phoneme_tokens: int = 510

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            model_dir=os.getenv("KOKORO_MODEL_DIR", cls.model_dir),
            voices_path=os.getenv("KOKORO_VOICES_PATH", cls.voices_path),
            vocab_path=os.getenv("KOKORO_VOCAB_PATH", cls.vocab_path),
            device=os.getenv("KOKORO_DEVICE", cls.device),
            max_num_seqs=int(os.getenv("KOKORO_MAX_NUM_SEQS", cls.max_num_seqs)),
            sample_rate=int(os.getenv("KOKORO_SAMPLE_RATE", cls.sample_rate)),
            max_phoneme_tokens=int(
                os.getenv("KOKORO_MAX_PHONEME_TOKENS", cls.max_phoneme_tokens)
            ),
        )
```

Create empty `kokoro_vllm/model/__init__.py`, `kokoro_vllm/frontend/__init__.py`, `kokoro_vllm/server/__init__.py`.

- [ ] **Step 4: Add the optional extra and pytest markers to `pyproject.toml`**

Add after the `[dependency-groups]` block:

```toml
[project.optional-dependencies]
vllm = [
    "vllm>=0.9.0",          # NOTE: must include V1 pooling + multimodal support; pin exact build after Task 2 verification
    "torch>=2.4",
    "kokoro>=0.9.4",
    "misaki>=0.9.4",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "numpy>=1.26",
    "soundfile>=0.13.0",
]

[tool.pytest.ini_options]
markers = [
    "gpu: tests that require a CUDA GPU and real model weights",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_scaffold.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add kokoro_vllm/ tests/test_scaffold.py pyproject.toml
git commit -m "feat(vllm): scaffold kokoro_vllm package and optional extra"
```

---

## Task 2: Pin vLLM & record its pooling/MM interface

**Files:**
- Create: `docs/superpowers/plans/vllm-interface-notes.md`
- Modify: `pyproject.toml:vllm` extra (replace `>=0.9.0` with the exact verified version)

This task has **no code test**; its deliverable is a verified, written record of the exact vLLM V1 interfaces the later tasks code against. It exists because vLLM's pooling/multimodal API moves between releases and every model task depends on these signatures.

- [ ] **Step 1: Install the extra into the working environment**

Run: `uv pip install -e ".[vllm]"`
Expected: vLLM + torch + kokoro install without error. Record the resolved `vllm` version:
Run: `python -c "import vllm; print(vllm.__version__)"`

- [ ] **Step 2: Capture the exact interface signatures**

Run and paste the output into `docs/superpowers/plans/vllm-interface-notes.md`:

```bash
python - <<'PY'
import inspect
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.layers.pooler import Pooler
from vllm import LLM
try:
    from vllm.v1.engine.async_llm import AsyncLLM
except Exception as e:
    AsyncLLM = None
    print("AsyncLLM import note:", e)

for obj in [SupportsMultiModal, VllmModelForPooling, Pooler]:
    print("====", obj.__name__, "====")
    print([m for m in dir(obj) if not m.startswith("__")])

print("==== Pooler abstract methods ====")
print(getattr(Pooler, "__abstractmethods__", None))
print("==== LLM.encode signature ====")
print(inspect.signature(LLM.encode))
if AsyncLLM is not None:
    print("==== AsyncLLM.encode signature ====")
    print(inspect.signature(AsyncLLM.encode))
PY
```

Record in the notes file, verbatim, the answers to:
1. `Pooler` required abstract methods (e.g. `get_supported_tasks`, `get_pooling_updates`, `forward`) and their signatures.
2. How a model declares itself a pooling model (base class / mixin name) in this version.
3. The `SupportsMultiModal` required methods (e.g. `get_multimodal_embeddings`, `get_input_embeddings`) and the multimodal registry decorators (`MULTIMODAL_REGISTRY.register_processor`, `@support_torch_compile`, input-mapper API).
4. The `.encode(...)` argument names for passing `prompt_token_ids` + `multi_modal_data` + a `PoolingParams`, and the `PoolingRequestOutput` field that holds the returned tensor (e.g. `.outputs.data`).

- [ ] **Step 3: Pin the exact version**

Edit `pyproject.toml`: replace `"vllm>=0.9.0"` with `"vllm==<resolved version>"` and drop the NOTE comment.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/vllm-interface-notes.md pyproject.toml
git commit -m "docs(vllm): pin vllm version and record V1 pooling/MM interfaces"
```

---

## Task 3: Phoneme vocab + `phonemes → input_ids`

**Files:**
- Create: `kokoro_vllm/frontend/vocab.py`
- Test: `tests/frontend/test_vocab.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_vocab(vocab_path: str) -> dict[str, int]` — reads Kokoro `config.json` `vocab` map.
  - `phonemes_to_input_ids(phonemes: str, vocab: dict[str, int]) -> list[int]` — maps each phoneme char via `vocab.get`, dropping unknowns, then pads with a leading and trailing `0` → `[0, *ids, 0]` (matches `KModel`).
  - `MAX_TOKENS = 510`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_vocab.py
import json
from kokoro_vllm.frontend.vocab import phonemes_to_input_ids, load_vocab, MAX_TOKENS

VOCAB = {"h": 5, "ɛ": 6, "l": 7, "o": 8}

def test_padding_and_mapping():
    ids = phonemes_to_input_ids("hɛllo", VOCAB)
    assert ids == [0, 5, 6, 7, 7, 8, 0]

def test_drops_unknown_phonemes():
    ids = phonemes_to_input_ids("hZo", VOCAB)  # Z not in vocab
    assert ids == [0, 5, 8, 0]

def test_max_tokens_constant():
    assert MAX_TOKENS == 510

def test_load_vocab(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"vocab": VOCAB}))
    assert load_vocab(str(p)) == VOCAB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/frontend/vocab.py
import json

MAX_TOKENS = 510


def load_vocab(vocab_path: str) -> dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["vocab"]


def phonemes_to_input_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab[p] for p in phonemes if p in vocab]
    return [0, *ids, 0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_vocab.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/frontend/vocab.py tests/frontend/test_vocab.py
git commit -m "feat(frontend): phoneme vocab load and input_id encoding"
```

---

## Task 4: G2P wrapper (misaki + espeak fallback)

**Files:**
- Create: `kokoro_vllm/frontend/g2p.py`
- Test: `tests/frontend/test_g2p.py`

**Interfaces:**
- Consumes: nothing (wraps `misaki`).
- Produces:
  - `class G2P:` with `__init__(self, lang: str = "en-us")` and `phonemize(self, text: str) -> str`.
  - `LANG_TO_MISAKI: dict[str, str]` mapping the CLI lang codes (`en-us`, `en-gb`, `fr-fr`, `it`, `ja`, `cmn`) to misaki backends; unsupported langs raise `ValueError`.

Note: misaki's real API is `misaki.espeak.EspeakG2P` / `misaki.en.G2P`. The G2P class must be a thin adapter so tests can monkeypatch the underlying callable. Do **not** hard-depend on network/model downloads in unit tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_g2p.py
import pytest
from kokoro_vllm.frontend import g2p as g2p_mod
from kokoro_vllm.frontend.g2p import G2P

def test_phonemize_delegates(monkeypatch):
    # Replace the backend factory so no model download happens
    class FakeBackend:
        def __call__(self, text):
            return ("hɛlo", None)   # misaki returns (phonemes, tokens)
    monkeypatch.setattr(g2p_mod, "_build_backend", lambda lang: FakeBackend())
    g = G2P(lang="en-us")
    assert g.phonemize("hello") == "hɛlo"

def test_unsupported_lang_raises():
    with pytest.raises(ValueError):
        G2P(lang="kl-fake")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_g2p.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/frontend/g2p.py
LANG_TO_MISAKI = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "fr-fr": "fr-fr",
    "it": "it",
    "ja": "ja",
    "cmn": "cmn",
}


def _build_backend(lang: str):
    # Imported lazily so unit tests can monkeypatch this factory.
    if lang in ("en-us", "en-gb"):
        from misaki import en
        british = lang == "en-gb"
        return en.G2P(british=british)
    from misaki import espeak
    return espeak.EspeakG2P(language=LANG_TO_MISAKI[lang])


class G2P:
    def __init__(self, lang: str = "en-us"):
        if lang not in LANG_TO_MISAKI:
            raise ValueError(
                f"Unsupported language: {lang}. "
                f"Supported: {', '.join(sorted(LANG_TO_MISAKI))}"
            )
        self.lang = lang
        self._backend = _build_backend(lang)

    def phonemize(self, text: str) -> str:
        phonemes, _ = self._backend(text)
        return phonemes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_g2p.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/frontend/g2p.py tests/frontend/test_g2p.py
git commit -m "feat(frontend): G2P wrapper over misaki with espeak fallback"
```

---

## Task 5: Chunker (≤510-token segments)

**Files:**
- Create: `kokoro_vllm/frontend/chunker.py`
- Test: `tests/frontend/test_chunker.py`

**Interfaces:**
- Consumes: `G2P.phonemize`, `phonemes_to_input_ids`, `MAX_TOKENS` from Task 3/4.
- Produces:
  - `chunk_text(text: str, g2p: G2P, vocab: dict[str,int], max_tokens: int = 510) -> list[Chunk]`
    where `Chunk` is a dataclass `Chunk(phonemes: str, input_ids: list[int])`.
  - Splits at sentence boundaries (`. ! ?` and newlines); if a single sentence exceeds `max_tokens` phonemes, splits on words; hard-truncates a single over-long word-run and logs a warning (never silently drops).
  - `input_ids` length (including the two padding zeros) never exceeds `max_tokens + 2`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_chunker.py
from kokoro_vllm.frontend.chunker import chunk_text, Chunk

class FakeG2P:
    # 1 phoneme char per input char, deterministic
    def phonemize(self, text):
        return text.replace(" ", "").replace(".", "")

# vocab maps every lowercase letter to a distinct id
VOCAB = {chr(c): c for c in range(ord("a"), ord("z") + 1)}

def test_short_text_single_chunk():
    chunks = chunk_text("ab. cd.", FakeG2P(), VOCAB, max_tokens=510)
    assert len(chunks) == 2                      # two sentences
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].input_ids[0] == 0 and chunks[0].input_ids[-1] == 0

def test_respects_max_tokens():
    long = " ".join(["abcde"] * 100) + "."       # 500 phoneme chars, one sentence
    chunks = chunk_text(long, FakeG2P(), VOCAB, max_tokens=120)
    for c in chunks:
        assert len(c.input_ids) <= 120 + 2

def test_never_empty_chunks():
    chunks = chunk_text("a.  . b.", FakeG2P(), VOCAB, max_tokens=510)
    assert all(len(c.input_ids) > 2 for c in chunks)   # more than just padding
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/frontend/chunker.py
import logging
import re
from dataclasses import dataclass

from kokoro_vllm.frontend.vocab import phonemes_to_input_ids

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")


@dataclass
class Chunk:
    phonemes: str
    input_ids: list[int]


def _token_len(phonemes: str, vocab: dict) -> int:
    # count of mappable phonemes (excludes the two padding zeros)
    return sum(1 for p in phonemes if p in vocab)


def _emit(phonemes: str, vocab: dict) -> Chunk | None:
    ids = phonemes_to_input_ids(phonemes, vocab)
    if len(ids) <= 2:  # only padding -> nothing real
        return None
    return Chunk(phonemes=phonemes, input_ids=ids)


def chunk_text(text, g2p, vocab, max_tokens=510):
    chunks: list[Chunk] = []
    for sentence in _SENTENCE_RE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        phon = g2p.phonemize(sentence)
        if _token_len(phon, vocab) <= max_tokens:
            c = _emit(phon, vocab)
            if c:
                chunks.append(c)
            continue
        # sentence too long: split on words
        buf = ""
        for word in sentence.split():
            wphon = g2p.phonemize(word)
            if _token_len(wphon, vocab) > max_tokens:
                logger.warning("Word exceeds max_tokens; hard-truncating: %r", word)
                wphon = "".join(p for p in wphon if p in vocab)[:max_tokens]
            candidate = (buf + " " + word).strip()
            cphon = g2p.phonemize(candidate)
            if _token_len(cphon, vocab) > max_tokens:
                c = _emit(g2p.phonemize(buf), vocab)
                if c:
                    chunks.append(c)
                buf = word
            else:
                buf = candidate
        c = _emit(g2p.phonemize(buf), vocab)
        if c:
            chunks.append(c)
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_chunker.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/frontend/chunker.py tests/frontend/test_chunker.py
git commit -m "feat(frontend): phoneme-token-aware chunker (<=510)"
```

---

## Task 6: Voice pack loader + `ref_s` selection + blending

**Files:**
- Create: `kokoro_vllm/frontend/voices.py`
- Test: `tests/frontend/test_voices.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `load_voicepacks(voices_path: str) -> dict[str, np.ndarray]` — each value shape `[510, 256]` (or `[510,1,256]` squeezed to `[510,256]`).
  - `parse_voice_spec(spec: str) -> list[tuple[str, float]]` — reuses the existing `"af_sarah:60,am_adam:40"` grammar; single voice → `[(name, 100.0)]`; normalizes weights to sum 100; raises `ValueError` on >2 voices or unknown syntax.
  - `select_ref_s(packs: dict, spec: str, num_tokens: int) -> np.ndarray` — resolves the blend, indexes each pack at `min(num_tokens, 509)`, weighted-sums, returns `float32[256]`.
  - `list_voices(packs) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_voices.py
import numpy as np
import pytest
from kokoro_vllm.frontend.voices import parse_voice_spec, select_ref_s, list_voices

def _packs():
    a = np.ones((510, 256), dtype=np.float32)
    b = np.full((510, 256), 3.0, dtype=np.float32)
    return {"af_sarah": a, "am_adam": b}

def test_parse_single():
    assert parse_voice_spec("af_sarah") == [("af_sarah", 100.0)]

def test_parse_blend_normalizes():
    out = parse_voice_spec("af_sarah:60,am_adam:40")
    assert out == [("af_sarah", 60.0), ("am_adam", 40.0)]

def test_parse_rejects_three():
    with pytest.raises(ValueError):
        parse_voice_spec("a,b,c")

def test_select_single_ref_s_shape_and_index():
    r = select_ref_s(_packs(), "af_sarah", num_tokens=5)
    assert r.shape == (256,)
    assert r.dtype == np.float32
    assert np.allclose(r, 1.0)

def test_select_blend_math():
    r = select_ref_s(_packs(), "af_sarah:50,am_adam:50", num_tokens=5)
    assert np.allclose(r, 0.5 * 1.0 + 0.5 * 3.0)  # == 2.0

def test_index_clamped():
    r = select_ref_s(_packs(), "af_sarah", num_tokens=9999)
    assert np.allclose(r, 1.0)  # clamped to 509, still valid

def test_list_voices():
    assert sorted(list_voices(_packs())) == ["af_sarah", "am_adam"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_voices.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/frontend/voices.py
import numpy as np


def load_voicepacks(voices_path: str) -> dict[str, np.ndarray]:
    # voices-v1.0.bin is a np.savez-style archive of {name: [510,(1,)256]}
    data = np.load(voices_path, allow_pickle=True)
    packs = {}
    for name in data.files:
        arr = np.asarray(data[name], dtype=np.float32)
        if arr.ndim == 3:            # [510,1,256] -> [510,256]
            arr = arr.squeeze(1)
        packs[name] = arr
    return packs


def list_voices(packs: dict[str, np.ndarray]) -> list[str]:
    return list(packs.keys())


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty voice spec")
    if len(parts) > 2:
        raise ValueError("Voice blending supports at most two voices")
    out = []
    for part in parts:
        if ":" in part:
            name, w = part.split(":")
            out.append((name.strip(), float(w.strip())))
        else:
            out.append((part, 50.0 if len(parts) == 2 else 100.0))
    total = sum(w for _, w in out)
    if len(parts) == 1:
        return [(out[0][0], 100.0)]
    return [(n, w * 100.0 / total) for n, w in out]


def select_ref_s(packs, spec, num_tokens):
    idx = min(max(num_tokens, 0), 509)
    blend = parse_voice_spec(spec)
    ref = np.zeros(256, dtype=np.float32)
    for name, weight in blend:
        if name not in packs:
            raise ValueError(
                f"Unsupported voice: {name}. "
                f"Available: {', '.join(sorted(packs))}"
            )
        ref += (weight / 100.0) * packs[name][idx]
    return ref.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_voices.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/frontend/voices.py tests/frontend/test_voices.py
git commit -m "feat(frontend): voicepack loading, ref_s selection, and blending"
```

---

## Task 7: HF-style config for the vLLM model

**Files:**
- Create: `kokoro_vllm/model/hf_config.py`
- Test: `tests/model/test_hf_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class KokoroConfig(PretrainedConfig)` with `model_type = "kokoro"`,
    `architectures = ["KokoroForConditionalGeneration"]`, and fields
    `hidden_dim`, `vocab_size`, `max_phoneme_len = 510`, `sample_rate = 24000`,
    plus the raw `vocab` dict and any `KModel` hyperparams read from Kokoro `config.json`.
  - `build_hf_config(kokoro_config_path: str) -> KokoroConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_hf_config.py
import json
from kokoro_vllm.model.hf_config import KokoroConfig, build_hf_config

def test_build(tmp_path):
    cfg = {"vocab": {"a": 1}, "hidden_dim": 512, "n_token": 178}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    c = build_hf_config(str(p))
    assert isinstance(c, KokoroConfig)
    assert c.architectures == ["KokoroForConditionalGeneration"]
    assert c.model_type == "kokoro"
    assert c.sample_rate == 24000
    assert c.max_phoneme_len == 510
    assert c.vocab == {"a": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_hf_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/model/hf_config.py
import json
from transformers import PretrainedConfig


class KokoroConfig(PretrainedConfig):
    model_type = "kokoro"

    def __init__(self, hidden_dim=512, vocab_size=178, max_phoneme_len=510,
                 sample_rate=24000, vocab=None, kmodel_kwargs=None, **kwargs):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_phoneme_len = max_phoneme_len
        self.sample_rate = sample_rate
        self.vocab = vocab or {}
        self.kmodel_kwargs = kmodel_kwargs or {}
        kwargs.setdefault("architectures", ["KokoroForConditionalGeneration"])
        super().__init__(**kwargs)


def build_hf_config(kokoro_config_path: str) -> KokoroConfig:
    with open(kokoro_config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return KokoroConfig(
        hidden_dim=raw.get("hidden_dim", 512),
        vocab_size=len(raw.get("vocab", {})) or raw.get("n_token", 178),
        vocab=raw.get("vocab", {}),
        kmodel_kwargs=raw,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_hf_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/model/hf_config.py tests/model/test_hf_config.py
git commit -m "feat(model): HF-style KokoroConfig"
```

---

## Task 8: KModel access + weight-name remap

**Files:**
- Create: `kokoro_vllm/model/kmodel_access.py`, `kokoro_vllm/model/weights.py`
- Create: `scripts/convert_weights.py`
- Test: `tests/model/test_weights_remap.py`

**Interfaces:**
- Consumes: `KokoroConfig` (Task 7).
- Produces:
  - `remap_param_name(kmodel_name: str) -> str` — pure string function mapping a raw `KModel` state-dict key to the name under our vLLM module tree (prefix `kmodel.`), so `load_weights` can match. Deterministic and unit-testable without weights.
  - `build_kmodel(config: KokoroConfig, device: str) -> torch.nn.Module` — instantiates the real `kokoro.KModel` submodules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) without downloading, given a local weights dir. (Wrapped so the GPU parity test in Task 12 can call it.)
  - `scripts/convert_weights.py`: CLI `python scripts/convert_weights.py --pth kokoro-v1_0.pth --out kokoro-model/` producing `model.safetensors` + `config.json`.

- [ ] **Step 1: Write the failing test** (pure remap, no weights/GPU)

```python
# tests/model/test_weights_remap.py
from kokoro_vllm.model.weights import remap_param_name

def test_prefixes_kmodel():
    assert remap_param_name("bert.encoder.layer.0.weight") == "kmodel.bert.encoder.layer.0.weight"

def test_predictor_and_decoder():
    assert remap_param_name("predictor.lstm.weight_ih_l0") == "kmodel.predictor.lstm.weight_ih_l0"
    assert remap_param_name("decoder.generator.conv.weight") == "kmodel.decoder.generator.conv.weight"

def test_idempotent_when_already_prefixed():
    assert remap_param_name("kmodel.bert.x") == "kmodel.bert.x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_weights_remap.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/model/weights.py
_PREFIX = "kmodel."


def remap_param_name(kmodel_name: str) -> str:
    if kmodel_name.startswith(_PREFIX):
        return kmodel_name
    return _PREFIX + kmodel_name
```

```python
# kokoro_vllm/model/kmodel_access.py
import torch


def build_kmodel(config, device: str) -> torch.nn.Module:
    """Instantiate the real kokoro.KModel from local files (no download)."""
    from kokoro import KModel
    # KModel accepts an explicit config path + model path when repo_id is None.
    model = KModel(
        config=config.kmodel_kwargs,
        model=None,          # weights loaded separately by vLLM load_weights
        disable_complex=False,
    ).to(device).eval()
    return model
```

```python
# scripts/convert_weights.py
import argparse
import json
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to kokoro-v1_0.pth")
    ap.add_argument("--config", required=True, help="Path to Kokoro config.json")
    ap.add_argument("--out", required=True, help="Output model dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state = torch.load(args.pth, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    # Flatten KModel's per-submodule dicts if present.
    flat = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    flat = {k: v.contiguous() for k, v in flat.items() if torch.is_tensor(v)}
    save_file(flat, os.path.join(args.out, "model.safetensors"))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["KokoroForConditionalGeneration"]
    cfg["model_type"] = "kokoro"
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"Wrote {args.out}/model.safetensors ({len(flat)} tensors)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_weights_remap.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/model/kmodel_access.py kokoro_vllm/model/weights.py scripts/convert_weights.py tests/model/test_weights_remap.py
git commit -m "feat(model): KModel access, weight remap, and .pth->safetensors converter"
```

---

## Task 9: Multimodal processor for `ref_s` + `speed`

**Files:**
- Create: `kokoro_vllm/model/mm_processor.py`
- Test: `tests/model/test_mm_processor.py`

**Interfaces:**
- Consumes: vLLM MM interfaces recorded in Task 2.
- Produces:
  - `normalize_ref_s(ref_s) -> torch.Tensor` — accepts list/ndarray/tensor, returns contiguous `float32[256]`; raises `ValueError` on wrong shape.
  - `class KokoroMultiModalProcessor` — the vLLM processor registered for modality `"voice"`; wraps a single `ref_s` item and threads `speed` through `mm_processor_kwargs`. The vLLM-coupled parts follow the signatures recorded in Task 2.

The **pure** `normalize_ref_s` helper is what we unit-test now (no vLLM/GPU); the processor class wiring is verified in the Task 12 GPU integration test.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_mm_processor.py
import numpy as np
import pytest
import torch
from kokoro_vllm.model.mm_processor import normalize_ref_s

def test_from_ndarray():
    r = normalize_ref_s(np.ones(256, dtype=np.float32))
    assert isinstance(r, torch.Tensor)
    assert r.shape == (256,)
    assert r.dtype == torch.float32

def test_from_list():
    r = normalize_ref_s([0.0] * 256)
    assert r.shape == (256,)

def test_bad_shape_raises():
    with pytest.raises(ValueError):
        normalize_ref_s(np.ones(128, dtype=np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_mm_processor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the pure helper (+ processor stub wired per Task 2 notes)**

```python
# kokoro_vllm/model/mm_processor.py
import numpy as np
import torch


def normalize_ref_s(ref_s) -> torch.Tensor:
    if isinstance(ref_s, torch.Tensor):
        t = ref_s.detach().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(ref_s, dtype=np.float32))
    t = t.reshape(-1)
    if t.shape != (256,):
        raise ValueError(f"ref_s must have 256 elements, got {tuple(t.shape)}")
    return t.contiguous()

# NOTE: The KokoroMultiModalProcessor class below must implement the exact
# base class + method names recorded in docs/superpowers/plans/vllm-interface-notes.md
# (Task 2). Wire register via MULTIMODAL_REGISTRY.register_processor in
# kokoro_vllm_model.py (Task 11). Keep normalize_ref_s as the single source of
# truth for ref_s validation inside the processor.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_mm_processor.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/model/mm_processor.py tests/model/test_mm_processor.py
git commit -m "feat(model): ref_s normalization helper for multimodal input"
```

---

## Task 10: Waveform pooler

**Files:**
- Create: `kokoro_vllm/model/pooler.py`
- Test: `tests/model/test_pooler.py`

**Interfaces:**
- Consumes: `Pooler` interface + `get_supported_tasks` signature from Task 2.
- Produces:
  - `class KokoroWaveformPooler(Pooler)` — returns each request's decoder waveform tensor as its pooled output (passthrough of per-request audio, no normalization). `get_supported_tasks()` returns `("embed",)` (the task the frontend submits under).
  - `_slice_per_request(flat_audio, seq_lengths) -> list[torch.Tensor]` — pure helper splitting a concatenated audio batch into per-request tensors; unit-tested without vLLM.

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_pooler.py
import torch
from kokoro_vllm.model.pooler import _slice_per_request

def test_slice_per_request():
    flat = torch.arange(10, dtype=torch.float32)
    out = _slice_per_request(flat, [3, 7])
    assert len(out) == 2
    assert torch.equal(out[0], torch.arange(0, 3, dtype=torch.float32))
    assert torch.equal(out[1], torch.arange(3, 10, dtype=torch.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_pooler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/model/pooler.py
import torch

# NOTE: base class import + method signatures per Task 2 notes.
from vllm.model_executor.layers.pooler import Pooler


def _slice_per_request(flat_audio: torch.Tensor, seq_lengths: list[int]) -> list[torch.Tensor]:
    out, start = [], 0
    for n in seq_lengths:
        out.append(flat_audio[start:start + n])
        start += n
    return out


class KokoroWaveformPooler(Pooler):
    """Passes each request's synthesized waveform through as its pooled output."""

    def get_supported_tasks(self):
        return ("embed",)

    def forward(self, hidden_states, pooling_metadata):
        # hidden_states here carries per-request audio produced by the model
        # forward (see kokoro_vllm_model.py). Split by the audio lengths the
        # model recorded in pooling_metadata and return the list.
        seq_lengths = pooling_metadata.audio_lengths  # set by the model forward
        if isinstance(hidden_states, (list, tuple)):
            return list(hidden_states)
        return _slice_per_request(hidden_states, seq_lengths)
```

Adjust the `forward` signature/return wrapper to match the exact `Pooler` contract recorded in Task 2 (e.g. wrapping in `PoolerOutput`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_pooler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/model/pooler.py tests/model/test_pooler.py
git commit -m "feat(model): waveform passthrough pooler"
```

---

## Task 11: `KokoroForConditionalGeneration` + registration

**Files:**
- Create: `kokoro_vllm/model/kokoro_vllm_model.py`, `kokoro_vllm/model/register.py`
- Modify: `pyproject.toml` (add `[project.entry-points."vllm.general_plugins"]`)
- Test: covered by the Task 12 GPU parity test (this task has no standalone CPU unit test because it requires the vLLM model runner + real weights).

**Interfaces:**
- Consumes: `build_kmodel`, `remap_param_name` (Task 8), `normalize_ref_s`/`KokoroMultiModalProcessor` (Task 9), `KokoroWaveformPooler` (Task 10), `KokoroConfig` (Task 7), and the vLLM base classes recorded in Task 2.
- Produces:
  - `class KokoroForConditionalGeneration(nn.Module, SupportsMultiModal, VllmModelForPooling)` with:
    - `__init__(self, *, vllm_config, prefix="")` — builds `self.kmodel = build_kmodel(...)`, `self.pooler = KokoroWaveformPooler()`.
    - `forward(self, input_ids, positions, intermediate_tensors=None, **kwargs)` — reads `ref_s` from `kwargs` (batched multimodal), calls `self.kmodel.forward_with_tokens(input_ids, ref_s, speed)` per request, returns per-request audio + records `audio_lengths` for the pooler.
    - `load_weights(self, weights)` — applies `remap_param_name` and loads into `self.kmodel`.
    - MM hooks (`get_multimodal_embeddings`, `get_input_embeddings`) per Task 2 signatures; `ref_s` is global conditioning (not placeholder-expanded), so `get_multimodal_embeddings` returns empty and `forward` consumes `ref_s` directly.
  - `register()` in `register.py` calling `ModelRegistry.register_model("KokoroForConditionalGeneration", KokoroForConditionalGeneration)` and `MULTIMODAL_REGISTRY.register_processor(...)`.

- [ ] **Step 1: Implement the model** (guided by Task 2 notes; exact base-class method names/decorators come from there)

```python
# kokoro_vllm/model/kokoro_vllm_model.py
import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.multimodal import MULTIMODAL_REGISTRY

from kokoro_vllm.model.hf_config import KokoroConfig
from kokoro_vllm.model.kmodel_access import build_kmodel
from kokoro_vllm.model.weights import remap_param_name
from kokoro_vllm.model.pooler import KokoroWaveformPooler
from kokoro_vllm.model.mm_processor import KokoroMultiModalProcessor


@MULTIMODAL_REGISTRY.register_processor(KokoroMultiModalProcessor)
class KokoroForConditionalGeneration(nn.Module, SupportsMultiModal, VllmModelForPooling):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        device = vllm_config.device_config.device
        self.config = cfg
        self.kmodel = build_kmodel(cfg, str(device))
        self.pooler = KokoroWaveformPooler()

    def get_multimodal_embeddings(self, **kwargs):
        # ref_s is global conditioning, not token-placeholder embeddings.
        return None

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        return self.kmodel.bert.embeddings(input_ids) \
            if hasattr(self.kmodel.bert, "embeddings") else None

    @torch.no_grad()
    def forward(self, input_ids, positions, intermediate_tensors=None, **kwargs):
        ref_s = kwargs["voice"]          # [B, 256] batched by vLLM MM
        speed = float(kwargs.get("speed", 1.0))
        audios, lengths = [], []
        # Split the flattened batch into per-request token runs using the
        # sequence metadata vLLM provides (see Task 2 notes for the exact
        # attribute carrying per-request lengths).
        for ids, rs in zip(_split_input_ids(input_ids, kwargs), ref_s):
            audio, _dur = self.kmodel.forward_with_tokens(
                ids.unsqueeze(0), rs.unsqueeze(0), speed
            )
            audio = audio.squeeze(0)
            audios.append(audio)
            lengths.append(audio.shape[-1])
        return _PooledAudio(audios=audios, lengths=lengths)

    def load_weights(self, weights):
        remapped = ((remap_param_name(name), w) for name, w in weights)
        params = dict(self.named_parameters())
        loaded = set()
        for name, w in remapped:
            if name in params:
                params[name].data.copy_(w)
                loaded.add(name)
        return loaded
```

Define the small `_split_input_ids` helper and `_PooledAudio` carrier consistent with the pooler's expected input (Task 10) and Task 2's sequence-metadata attribute. Wire `self.pooler.forward` to read `_PooledAudio`.

- [ ] **Step 2: Implement registration**

```python
# kokoro_vllm/model/register.py
from vllm import ModelRegistry


def register():
    from kokoro_vllm.model.kokoro_vllm_model import KokoroForConditionalGeneration
    ModelRegistry.register_model(
        "KokoroForConditionalGeneration",
        KokoroForConditionalGeneration,
    )
```

- [ ] **Step 3: Add the plugin entry-point to `pyproject.toml`**

```toml
[project.entry-points."vllm.general_plugins"]
kokoro_register = "kokoro_vllm.model.register:register"
```

- [ ] **Step 4: Verify the module imports under the installed vLLM**

Run: `python -c "import kokoro_vllm.model.kokoro_vllm_model as m; print(m.KokoroForConditionalGeneration.__name__)"`
Expected: prints `KokoroForConditionalGeneration` with no import error. Fix method names/imports against Task 2 notes until it imports cleanly.

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/model/kokoro_vllm_model.py kokoro_vllm/model/register.py pyproject.toml
git commit -m "feat(model): KokoroForConditionalGeneration vLLM model + plugin registration"
```

---

## Task 12: GPU parity test (model runs on the engine)

**Files:**
- Create: `tests/model/test_parity_gpu.py`
- Requires: a CUDA GPU + converted weights (`scripts/convert_weights.py` output) + `voices-v1.0.bin` + Kokoro `config.json`. Skips cleanly when absent.

**Interfaces:**
- Consumes: the full model stack (Tasks 3–11) + `kokoro-onnx`/reference `KModel` for the reference waveform.
- Produces: proof that identical `(phonemes, ref_s, speed)` through the **vLLM engine** matches the reference `KModel` waveform within tolerance.

- [ ] **Step 1: Write the GPU test**

```python
# tests/model/test_parity_gpu.py
import os
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

MODEL_DIR = os.getenv("KOKORO_MODEL_DIR", "./kokoro-model")
VOICES = os.getenv("KOKORO_VOICES_PATH", "./voices-v1.0.bin")
VOCAB = os.getenv("KOKORO_VOCAB_PATH", "./kokoro-model/config.json")

@pytest.mark.skipif(not os.path.exists(MODEL_DIR), reason="no converted weights")
def test_vllm_matches_reference_kmodel():
    from vllm import LLM, PoolingParams
    from kokoro_vllm.frontend.vocab import load_vocab, phonemes_to_input_ids
    from kokoro_vllm.frontend.voices import load_voicepacks, select_ref_s

    vocab = load_vocab(VOCAB)
    packs = load_voicepacks(VOICES)
    phonemes = "hˈɛlˌoʊ"                     # fixed phoneme string
    ids = phonemes_to_input_ids(phonemes, vocab)
    ref_s = select_ref_s(packs, "af_sarah", num_tokens=len(ids) - 2)

    # --- vLLM engine path ---
    llm = LLM(model=MODEL_DIR, task="embed", enforce_eager=True,
              trust_remote_code=True)
    out = llm.encode({
        "prompt_token_ids": ids,
        "multi_modal_data": {"voice": ref_s},
    }, pooling_params=PoolingParams(task="embed"))
    vllm_audio = np.asarray(out[0].outputs.data).reshape(-1)

    # --- reference KModel path ---
    from kokoro import KModel
    import torch
    km = KModel(config=VOCAB, model=os.path.join(MODEL_DIR, "model.safetensors")).eval()
    ref_audio, _ = km.forward_with_tokens(
        torch.tensor([ids]), torch.tensor(ref_s).unsqueeze(0), 1.0)
    ref_audio = ref_audio.squeeze(0).detach().cpu().numpy()

    n = min(len(vllm_audio), len(ref_audio))
    corr = np.corrcoef(vllm_audio[:n], ref_audio[:n])[0, 1]
    assert corr > 0.99, f"waveform correlation too low: {corr}"
```

- [ ] **Step 2: Run the GPU test**

Run: `pytest tests/model/test_parity_gpu.py -v -m gpu`
Expected: PASS on a GPU host with weights present; SKIP otherwise. If correlation fails, debug the `load_weights` remap and the `forward` per-request splitting against Task 2 notes before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_parity_gpu.py
git commit -m "test(model): GPU parity between vLLM engine and reference KModel"
```

---

## Task 13: Async engine lifecycle wrapper

**Files:**
- Create: `kokoro_vllm/server/engine.py`
- Test: `tests/server/test_engine.py`

**Interfaces:**
- Consumes: `ServerSettings` (Task 1), the vLLM `AsyncLLM.encode` signature (Task 2).
- Produces:
  - `class KokoroEngine` with `async def start(self)`, `async def synthesize_chunk(self, input_ids: list[int], ref_s: np.ndarray, speed: float, request_id: str) -> np.ndarray`, `async def close(self)`, and `def is_ready(self) -> bool`.
  - `synthesize_chunk` submits one `AsyncLLM.encode` request and returns the chunk's `float32` waveform.

The unit test drives a **fake engine** injected via constructor so no GPU is needed; the real `AsyncLLM` wiring is exercised by Task 12/16.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_engine.py
import asyncio
import numpy as np
from kokoro_vllm.server.engine import KokoroEngine

class FakeAsyncLLM:
    async def encode(self, prompt, pooling_params=None, request_id=None):
        # mimic vLLM async generator yielding a final PoolingRequestOutput-like obj
        class Out:
            class outputs: data = np.ones(1200, dtype=np.float32)
        yield Out()

def test_synthesize_chunk_returns_audio():
    eng = KokoroEngine(settings=None, _llm=FakeAsyncLLM())
    audio = asyncio.run(eng.synthesize_chunk([0, 5, 0], np.ones(256, np.float32), 1.0, "r1"))
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (1200,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/server/engine.py
import numpy as np


class KokoroEngine:
    def __init__(self, settings, _llm=None):
        self.settings = settings
        self._llm = _llm
        self._ready = _llm is not None

    async def start(self):
        if self._llm is None:
            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm.engine.arg_utils import AsyncEngineArgs
            args = AsyncEngineArgs(
                model=self.settings.model_dir,
                task="embed",
                max_num_seqs=self.settings.max_num_seqs,
                trust_remote_code=True,
            )
            self._llm = AsyncLLM.from_engine_args(args)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def synthesize_chunk(self, input_ids, ref_s, speed, request_id):
        from vllm import PoolingParams
        prompt = {
            "prompt_token_ids": list(input_ids),
            "multi_modal_data": {"voice": np.asarray(ref_s, dtype=np.float32)},
        }
        final = None
        async for out in self._llm.encode(
            prompt,
            pooling_params=PoolingParams(task="embed"),
            request_id=request_id,
        ):
            final = out
        return np.asarray(final.outputs.data, dtype=np.float32).reshape(-1)

    async def close(self):
        if self._llm is not None and hasattr(self._llm, "shutdown"):
            self._llm.shutdown()
        self._ready = False
```

Confirm `AsyncLLM.encode`'s exact kwargs (`pooling_params`, `request_id`) against Task 2 notes; adjust if the installed version differs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/server/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/server/engine.py tests/server/test_engine.py
git commit -m "feat(server): async engine wrapper over vLLM AsyncLLM"
```

---

## Task 14: Request/response schemas

**Files:**
- Create: `kokoro_vllm/server/schemas.py`
- Test: `tests/server/test_schemas.py`

**Interfaces:**
- Produces:
  - `class SpeechRequest(BaseModel)`: `model: str = "kokoro"`, `input: str`, `voice: str = "af_sarah"`, `response_format: Literal["pcm","wav","mp3","opus"] = "wav"`, `speed: float = 1.0`, `lang: str = "en-us"`, `stream: bool = False`. Validates `0.5 <= speed <= 2.0` and non-empty `input`.
  - `class VoicesResponse(BaseModel)`: `voices: list[str]`, `languages: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_schemas.py
import pytest
from pydantic import ValidationError
from kokoro_vllm.server.schemas import SpeechRequest

def test_defaults():
    r = SpeechRequest(input="hi")
    assert r.voice == "af_sarah"
    assert r.response_format == "wav"
    assert r.speed == 1.0

def test_rejects_empty_input():
    with pytest.raises(ValidationError):
        SpeechRequest(input="")

def test_rejects_bad_speed():
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", speed=9.0)

def test_rejects_bad_format():
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", response_format="flac")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/server/schemas.py
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str = Field(min_length=1)
    voice: str = "af_sarah"
    response_format: Literal["pcm", "wav", "mp3", "opus"] = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lang: str = "en-us"
    stream: bool = False

    @field_validator("input")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input must not be blank")
        return v


class VoicesResponse(BaseModel):
    voices: list[str]
    languages: list[str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/server/test_schemas.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/server/schemas.py tests/server/test_schemas.py
git commit -m "feat(server): request/response schemas"
```

---

## Task 15: Streaming — ordered fan-out + audio encoding

**Files:**
- Create: `kokoro_vllm/server/streaming.py`
- Test: `tests/server/test_streaming.py`

**Interfaces:**
- Consumes: `KokoroEngine.synthesize_chunk` (Task 13), `Chunk` (Task 5), `select_ref_s` (Task 6).
- Produces:
  - `encode_audio(samples: np.ndarray, fmt: str, sample_rate: int) -> bytes` — pcm (raw int16 LE), wav (via `soundfile`), mp3/opus (via `ffmpeg` subprocess); raises `AudioEncodeError` (subclass of `ValueError`) if `ffmpeg` missing for mp3/opus.
  - `async def stream_synthesis(engine, chunks, packs, voice, speed, fmt, sample_rate) -> AsyncIterator[bytes]` — submits **all** chunks concurrently, yields encoded bytes **in submission order** (buffer out-of-order completions).

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_streaming.py
import asyncio
import numpy as np
import pytest
from kokoro_vllm.server.streaming import encode_audio, stream_synthesis, AudioEncodeError
from kokoro_vllm.frontend.chunker import Chunk

def test_encode_pcm_roundtrip():
    samples = np.zeros(100, dtype=np.float32)
    b = encode_audio(samples, "pcm", 24000)
    assert isinstance(b, bytes)
    assert len(b) == 200          # 100 int16 samples

def test_encode_wav_has_riff_header():
    b = encode_audio(np.zeros(100, np.float32), "wav", 24000)
    assert b[:4] == b"RIFF"

class FakeEngine:
    async def synthesize_chunk(self, input_ids, ref_s, speed, request_id):
        # later chunks finish first, to prove ordering is enforced
        await asyncio.sleep(0.02 if input_ids[1] == 1 else 0.0)
        return np.full(input_ids[1] * 10, input_ids[1], dtype=np.float32)

def test_stream_preserves_order():
    packs = {"af_sarah": np.ones((510, 256), np.float32)}
    chunks = [Chunk("x", [0, 1, 0]), Chunk("y", [0, 2, 0])]

    async def run():
        out = []
        async for b in stream_synthesis(FakeEngine(), chunks, packs,
                                        "af_sarah", 1.0, "pcm", 24000):
            out.append(b)
        return out

    out = asyncio.run(run())
    # chunk 0 (value 1) must come before chunk 1 (value 2) despite finishing later
    first = np.frombuffer(out[0], dtype=np.int16)
    assert first[0] == 1

def test_mp3_without_ffmpeg(monkeypatch):
    import kokoro_vllm.server.streaming as s
    monkeypatch.setattr(s, "_have_ffmpeg", lambda: False)
    with pytest.raises(AudioEncodeError):
        encode_audio(np.zeros(100, np.float32), "mp3", 24000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server/test_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/server/streaming.py
import asyncio
import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

from kokoro_vllm.frontend.voices import select_ref_s


class AudioEncodeError(ValueError):
    pass


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _to_int16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def encode_audio(samples: np.ndarray, fmt: str, sample_rate: int) -> bytes:
    if fmt == "pcm":
        return _to_int16(samples).tobytes()
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    if fmt in ("mp3", "opus"):
        if not _have_ffmpeg():
            raise AudioEncodeError(
                f"{fmt} requires ffmpeg; install ffmpeg or use pcm/wav"
            )
        codec = "libmp3lame" if fmt == "mp3" else "libopus"
        proc = subprocess.run(
            ["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
             "-i", "pipe:0", "-f", fmt, "-c:a", codec, "pipe:1"],
            input=_to_int16(samples).tobytes(),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        )
        return proc.stdout
    raise AudioEncodeError(f"Unknown format: {fmt}")


async def stream_synthesis(engine, chunks, packs, voice, speed, fmt, sample_rate):
    async def run(i, chunk):
        num_tokens = len(chunk.input_ids) - 2
        ref_s = select_ref_s(packs, voice, num_tokens)
        audio = await engine.synthesize_chunk(
            chunk.input_ids, ref_s, speed, request_id=f"req-{id(chunks)}-{i}")
        return i, audio

    tasks = [asyncio.ensure_future(run(i, c)) for i, c in enumerate(chunks)]
    pending = {t for t in tasks}
    buffered: dict[int, np.ndarray] = {}
    next_idx = 0
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            i, audio = t.result()
            buffered[i] = audio
        while next_idx in buffered:
            yield encode_audio(buffered.pop(next_idx), fmt, sample_rate)
            next_idx += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/server/test_streaming.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/server/streaming.py tests/server/test_streaming.py
git commit -m "feat(server): ordered streaming fan-out and audio encoding"
```

---

## Task 16: FastAPI app + routes + error handling

**Files:**
- Create: `kokoro_vllm/server/app.py`
- Test: `tests/server/test_app.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `def create_app(engine, g2p_factory, vocab, packs, settings) -> FastAPI` — dependency-injected so tests use a fake engine/G2P; no GPU needed.
  - Routes: `POST /v1/audio/speech`, `GET /v1/audio/voices`, `GET /health`.
  - `POST /v1/audio/speech`: builds chunks (Task 5), validates voice (Task 6 `parse_voice_spec`), and either streams (`StreamingResponse`, correct media type per format) or returns the concatenated buffer.
  - Error mapping: `ValueError` → 400; engine-not-ready → 503; `AudioEncodeError` → 400.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_app.py
import numpy as np
from fastapi.testclient import TestClient
from kokoro_vllm.server.app import create_app

class FakeEngine:
    def is_ready(self): return True
    async def synthesize_chunk(self, input_ids, ref_s, speed, request_id):
        return np.zeros(240, dtype=np.float32)

class FakeG2P:
    def phonemize(self, text): return text.replace(" ", "").replace(".", "")

VOCAB = {chr(c): c for c in range(ord("a"), ord("z") + 1)}
PACKS = {"af_sarah": np.ones((510, 256), np.float32)}

class Settings:
    sample_rate = 24000

def _client():
    app = create_app(FakeEngine(), lambda lang: FakeG2P(), VOCAB, PACKS, Settings())
    return TestClient(app)

def test_health_ok():
    assert _client().get("/health").json()["status"] == "ok"

def test_voices_lists():
    body = _client().get("/v1/audio/voices").json()
    assert "af_sarah" in body["voices"]

def test_speech_wav_non_stream():
    r = _client().post("/v1/audio/speech",
                       json={"input": "ab. cd.", "voice": "af_sarah",
                             "response_format": "wav"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"

def test_speech_bad_voice_400():
    r = _client().post("/v1/audio/speech",
                       json={"input": "ab.", "voice": "nope"})
    assert r.status_code == 400

def test_speech_empty_input_422():
    r = _client().post("/v1/audio/speech", json={"input": ""})
    assert r.status_code == 422    # schema validation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# kokoro_vllm/server/app.py
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from kokoro_vllm.frontend.chunker import chunk_text
from kokoro_vllm.frontend.voices import list_voices, parse_voice_spec
from kokoro_vllm.frontend.g2p import LANG_TO_MISAKI
from kokoro_vllm.server.schemas import SpeechRequest, VoicesResponse
from kokoro_vllm.server.streaming import (
    encode_audio, stream_synthesis, AudioEncodeError,
)

_MEDIA = {"pcm": "audio/pcm", "wav": "audio/wav",
          "mp3": "audio/mpeg", "opus": "audio/opus"}


def create_app(engine, g2p_factory, vocab, packs, settings) -> FastAPI:
    app = FastAPI(title="Kokoro-vLLM")

    @app.get("/health")
    def health():
        return {"status": "ok" if engine.is_ready() else "unready"}

    @app.get("/v1/audio/voices", response_model=VoicesResponse)
    def voices():
        return VoicesResponse(voices=sorted(list_voices(packs)),
                              languages=sorted(LANG_TO_MISAKI))

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not engine.is_ready():
            raise HTTPException(status_code=503, detail="engine not ready")
        try:
            parse_voice_spec(req.voice)          # validate early
            for name, _ in parse_voice_spec(req.voice):
                if name not in packs:
                    raise ValueError(f"Unsupported voice: {name}")
            g2p = g2p_factory(req.lang)
            chunks = chunk_text(req.input, g2p, vocab)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not chunks:
            raise HTTPException(status_code=400, detail="no synthesizable text")

        media = _MEDIA[req.response_format]
        agen = stream_synthesis(engine, chunks, packs, req.voice, req.speed,
                                req.response_format, settings.sample_rate)
        if req.stream:
            return StreamingResponse(agen, media_type=media)
        try:
            body = b"".join([b async for b in agen])
        except AudioEncodeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=body, media_type=media)

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/server/test_app.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add kokoro_vllm/server/app.py tests/server/test_app.py
git commit -m "feat(server): FastAPI app with speech/voices/health routes"
```

---

## Task 17: Server entry-point + README section + full non-GPU suite

**Files:**
- Create: `kokoro_vllm/server/__main__.py`
- Modify: `README.md` (add a "vLLM Streaming Server" section)
- Test: run the whole non-GPU suite green.

**Interfaces:**
- Produces: `python -m kokoro_vllm.server` launches uvicorn, wiring the real `KokoroEngine`, `G2P`, vocab, and voicepacks from `ServerSettings.from_env()`.

- [ ] **Step 1: Implement the entry-point**

```python
# kokoro_vllm/server/__main__.py
import asyncio
import uvicorn

from kokoro_vllm.config import ServerSettings
from kokoro_vllm.frontend.g2p import G2P
from kokoro_vllm.frontend.vocab import load_vocab
from kokoro_vllm.frontend.voices import load_voicepacks
from kokoro_vllm.server.engine import KokoroEngine
from kokoro_vllm.server.app import create_app


def build():
    settings = ServerSettings.from_env()
    engine = KokoroEngine(settings)
    asyncio.get_event_loop().run_until_complete(engine.start())
    vocab = load_vocab(settings.vocab_path)
    packs = load_voicepacks(settings.voices_path)
    return create_app(engine, lambda lang: G2P(lang), vocab, packs, settings)


if __name__ == "__main__":
    uvicorn.run(build(), host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Add a README section**

Add under a new `## vLLM Streaming Server` heading in `README.md`: install (`pip install kokoro-tts[vllm]`), convert weights (`python scripts/convert_weights.py --pth kokoro-v1_0.pth --config config.json --out kokoro-model/`), run (`python -m kokoro_vllm.server`), and a `curl` example hitting `/v1/audio/speech` with a streamed `wav`.

- [ ] **Step 3: Run the full non-GPU suite**

Run: `pytest -m "not gpu" -v`
Expected: all frontend/model(cpu)/server tests PASS.

- [ ] **Step 4: Commit**

```bash
git add kokoro_vllm/server/__main__.py README.md
git commit -m "feat(server): uvicorn entry-point and docs"
```

---

## Self-Review Notes (against the spec)

- **§3 Approach A / whole model on vLLM** → Tasks 7–12. ✅
- **§5.1 ref_s as multimodal** → Tasks 9, 11. ✅
- **§5.2 forward pass (forward_with_tokens)** → Task 11. ✅
- **§5.3 waveform as pooled output** → Task 10. ✅
- **§5.4 config + weights + voice pack in frontend** → Tasks 6, 7, 8. ✅
- **§5.5 registration plugin** → Task 11. ✅
- **§6 G2P / chunking / voice / streaming** → Tasks 4, 5, 6, 15. ✅
- **§7 API (speech/voices/health, formats)** → Tasks 14, 15, 16. ✅
- **§9 error handling** (400/503/ffmpeg/disconnect) → Tasks 15, 16 (disconnect handled by `StreamingResponse` cancellation; note in Task 16). ✅
- **§10 testing (unit/model-parity/server)** → Tasks 3–6, 8–10, 12, 13–16. ✅
- **§2.1 move off ONNX** → PyTorch `KModel` used throughout Tasks 8, 11; ONNX CLI untouched (Global Constraints). ✅
- **Version sensitivity** of vLLM APIs → Task 2 pins & records; Tasks 9–13 reference those notes. ✅

Gap intentionally deferred (per spec §7 "out of scope"): EPUB/PDF CLI → server wiring. Not planned here.
