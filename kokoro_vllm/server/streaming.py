import asyncio
import io
import shutil
import subprocess
import uuid

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
    base = uuid.uuid4().hex

    async def run(i, chunk):
        num_tokens = len(chunk.input_ids) - 2
        ref_s = select_ref_s(packs, voice, num_tokens)
        audio = await engine.synthesize_chunk(
            chunk.input_ids, ref_s, speed, request_id=f"req-{base}-{i}")
        return i, audio

    tasks = [asyncio.ensure_future(run(i, c)) for i, c in enumerate(chunks)]
    try:
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
    finally:
        # On any exit path — normal completion, an exception from a chunk
        # task, or GeneratorExit from the consumer closing this generator
        # early (client disconnect) — cancel any tasks still in flight so we
        # don't leave orphaned engine.synthesize_chunk calls running with no
        # consumer (frees the corresponding vLLM slots). Awaiting with
        # return_exceptions=True lets cancellation settle without raising.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
