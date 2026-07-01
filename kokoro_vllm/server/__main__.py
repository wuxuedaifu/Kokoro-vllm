"""Entry-point: `python -m kokoro_vllm.server` boots a real Kokoro TTS server.

Wires the real components — `KokoroEngine`, `G2P`, the phoneme vocab, and the
voicepacks — from `ServerSettings.from_env()`, and serves the FastAPI app
(`kokoro_vllm.server.app.create_app`) via uvicorn on 0.0.0.0:8000.

Engine lifecycle: the vLLM `AsyncLLM` engine is expensive to construct (it
loads the model onto the GPU) and must be torn down cleanly on shutdown. This
uses a FastAPI/Starlette **lifespan** context manager — `await engine.start()`
before the `yield`, `await engine.close()` after — instead of calling
`asyncio.get_event_loop().run_until_complete(engine.start())` at import time.
That pattern (see the task brief's illustrative snippet) is fragile: outside
a running event loop `get_event_loop()` is deprecated and may create a loop
that is never the one uvicorn actually serves on, and there is no defined
hook to run cleanup on shutdown. The lifespan runs inside uvicorn's real
event loop and gives us a paired startup/shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from kokoro_vllm.config import ServerSettings
from kokoro_vllm.frontend.g2p import G2P
from kokoro_vllm.frontend.vocab import load_vocab
from kokoro_vllm.frontend.voices import load_voicepacks
from kokoro_vllm.server.app import create_app
from kokoro_vllm.server.engine import KokoroEngine

logger = logging.getLogger("kokoro_vllm.server")


def build_app(settings: ServerSettings | None = None) -> FastAPI:
    """Construct the FastAPI app with a real engine, vocab, and voicepacks.

    Vocab and voicepacks are cheap, synchronous, file-backed loads, so they
    happen eagerly here. The engine itself is only *constructed* here;
    `engine.start()` (the expensive GPU-bound step) is deferred to the
    lifespan's startup phase so it runs on uvicorn's event loop.
    """
    settings = settings or ServerSettings.from_env()
    engine = KokoroEngine(settings)
    vocab = load_vocab(settings.vocab_path)
    packs = load_voicepacks(settings.voices_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Kokoro engine (model_dir=%s)...", settings.model_dir)
        await engine.start()
        logger.info("Kokoro engine ready.")
        try:
            yield
        finally:
            logger.info("Shutting down Kokoro engine...")
            await engine.close()

    return create_app(engine, lambda lang: G2P(lang), vocab, packs, settings,
                       lifespan=lifespan)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
