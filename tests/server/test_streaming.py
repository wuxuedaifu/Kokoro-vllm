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
        # Note: keep the "identity" value within [-1, 1] (0.1 per chunk index)
        # so it survives encode_audio's clip-to-audio-range step and stays
        # distinguishable between chunks (see task-15-report.md for why the
        # brief's original raw 1/2 values collide after clipping).
        return np.full(input_ids[1] * 10, input_ids[1] * 0.1, dtype=np.float32)

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
    # chunk 0 (value 0.1) must come before chunk 1 (value 0.2) despite finishing later
    first = np.frombuffer(out[0], dtype=np.int16)
    second = np.frombuffer(out[1], dtype=np.int16)
    assert first[0] > 0
    assert first[0] < second[0]

def test_mp3_without_ffmpeg(monkeypatch):
    import kokoro_vllm.server.streaming as s
    monkeypatch.setattr(s, "_have_ffmpeg", lambda: False)
    with pytest.raises(AudioEncodeError):
        encode_audio(np.zeros(100, np.float32), "mp3", 24000)
