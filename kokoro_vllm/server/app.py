import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from kokoro_vllm.frontend.chunker import chunk_text
from kokoro_vllm.frontend.voices import list_voices, parse_voice_spec
from kokoro_vllm.frontend.g2p import LANG_TO_MISAKI
from kokoro_vllm.server.schemas import SpeechRequest, VoicesResponse
from kokoro_vllm.server import streaming
from kokoro_vllm.server.streaming import stream_synthesis, AudioEncodeError

_MEDIA = {"pcm": "audio/pcm", "wav": "audio/wav",
          "mp3": "audio/mpeg", "opus": "audio/opus"}


def create_app(engine, g2p_factory, vocab, packs, settings, lifespan=None) -> FastAPI:
    app = FastAPI(title="Kokoro-vLLM", lifespan=lifespan)

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

        if req.response_format in ("mp3", "opus") and not streaming._have_ffmpeg():
            raise HTTPException(
                status_code=400,
                detail=f"{req.response_format} requires ffmpeg; install ffmpeg or use pcm/wav",
            )

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
