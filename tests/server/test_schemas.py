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
