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


class CancelTrackingEngine:
    """Records which request_ids started, and which were cancelled mid-flight."""

    def __init__(self):
        self.started = []
        self.cancelled = set()

    async def synthesize_chunk(self, input_ids, ref_s, speed, request_id):
        self.started.append(request_id)
        # chunk 0 finishes fast; all later chunks sleep "long" so they are
        # still pending when the consumer breaks out early.
        idx = input_ids[1]
        try:
            if idx == 0:
                await asyncio.sleep(0.0)
            else:
                await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            self.cancelled.add(request_id)
            raise
        return np.full(10, 0.1, dtype=np.float32)


def test_stream_cancels_pending_on_early_exit():
    packs = {"af_sarah": np.ones((510, 256), np.float32)}
    chunks = [Chunk("a", [0, 0, 0]), Chunk("b", [0, 1, 0]), Chunk("c", [0, 2, 0])]
    engine = CancelTrackingEngine()

    async def run():
        agen = stream_synthesis(engine, chunks, packs, "af_sarah", 1.0, "pcm", 24000)
        out = []
        async for b in agen:
            out.append(b)
            break
        await agen.aclose()
        # Capture state immediately after aclose() returns, *inside* the same
        # coroutine that asyncio.run() drives. If stream_synthesis correctly
        # cancels-and-awaits its pending tasks in a finally block, cancellation
        # is already reflected here. We deliberately snapshot now (not after
        # run() returns) so the assertion isn't masked by asyncio.run()'s own
        # end-of-run task cleanup, which would cancel any leaked/orphaned
        # tasks anyway and produce a false pass.
        return len(out), set(engine.cancelled), set(engine.started)

    n, cancelled, started = asyncio.run(run())
    assert n == 1
    assert len(started) >= 2  # chunk 0 plus at least one still-pending chunk
    assert cancelled, "expected at least one pending chunk task to be cancelled " \
        "by the time stream_synthesis's async generator finished closing"
