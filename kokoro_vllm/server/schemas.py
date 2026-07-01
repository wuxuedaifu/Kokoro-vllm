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
