"""Real end-to-end HTTP smoke test: boots a REAL KokoroEngine on a REAL GPU,
builds the real FastAPI app (real vocab/voicepacks/G2P) via the same
`__main__.build_app` used by `python -m kokoro_vllm.server`, and drives it
with an HTTP test client — no fakes anywhere in this path.

This is the capstone check that the whole stack (uvicorn entry-point
lifespan wiring, `KokoroEngine.start()`, the FastAPI routes, the frontend
G2P/vocab/voice pipeline, and audio encoding) works together against the
real converted weights, not just against the individually-mocked unit tests.

Deliberately reuses `build_app()` (not a hand-rolled engine.start()/close())
so this test exercises exactly the lifespan-managed startup/shutdown path a
real deployment goes through: `TestClient(app)` used as a context manager
drives the ASGI lifespan protocol the same way uvicorn does, so
`engine.start()`/`engine.close()` and every request run on the *same* event
loop (avoiding cross-loop AsyncLLM background-task issues that a manual
`asyncio.run(engine.start())` + separate `TestClient` block would risk).

Skips cleanly when `kokoro-model/model.safetensors` is absent (no GPU / not
staged) so it never blocks the non-GPU suite. Requires a single visible CUDA
GPU: run with `CUDA_VISIBLE_DEVICES=0`. Booting the real engine takes a
couple of minutes.
"""

import io
import os

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu

MODEL_DIR = os.getenv("KOKORO_MODEL_DIR", "./kokoro-model")
VOICES = os.getenv("KOKORO_VOICES_PATH", "./voices-v1.0.bin")
VOCAB = os.getenv("KOKORO_VOCAB_PATH", "./kokoro-model/config.json")

SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")


@pytest.mark.skipif(
    not os.path.exists(SAFETENSORS),
    reason="no converted weights (run scripts/convert_weights.py)",
)
def test_real_engine_speech_endpoint_end_to_end():
    from kokoro_vllm.config import ServerSettings
    from kokoro_vllm.server.__main__ import build_app

    settings = ServerSettings(model_dir=MODEL_DIR, voices_path=VOICES, vocab_path=VOCAB)
    app = build_app(settings)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        voices_resp = client.get("/v1/audio/voices")
        assert voices_resp.status_code == 200
        assert "af_sarah" in voices_resp.json()["voices"]

        speech = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello world.",
                "voice": "af_sarah",
                "response_format": "wav",
                "stream": False,
            },
        )
        assert speech.status_code == 200
        assert speech.content[:4] == b"RIFF"

        samples, sr = sf.read(io.BytesIO(speech.content), dtype="float32")
        assert sr == 24000
        assert len(samples) > 0
        print(
            f"\n[e2e] wav bytes={len(speech.content)} samples={len(samples)} "
            f"duration_s={len(samples) / sr:.3f}"
        )
