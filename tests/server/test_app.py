import numpy as np
from fastapi.testclient import TestClient

from kokoro_vllm.server.app import create_app


class FakeEngine:
    def is_ready(self):
        return True

    async def synthesize_chunk(self, input_ids, ref_s, speed, request_id):
        return np.zeros(240, dtype=np.float32)


class FakeG2P:
    def phonemize(self, text):
        return text.replace(" ", "").replace(".", "")


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


def test_speech_mp3_without_ffmpeg_400(monkeypatch):
    monkeypatch.setattr("kokoro_vllm.server.streaming._have_ffmpeg", lambda: False)

    r = _client().post("/v1/audio/speech",
                       json={"input": "ab. cd.", "voice": "af_sarah",
                             "response_format": "mp3", "stream": True})
    assert r.status_code == 400

    r = _client().post("/v1/audio/speech",
                       json={"input": "ab. cd.", "voice": "af_sarah",
                             "response_format": "mp3", "stream": False})
    assert r.status_code == 400
