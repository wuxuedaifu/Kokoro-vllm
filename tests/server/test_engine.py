import asyncio

import numpy as np

from kokoro_vllm.server.engine import KokoroEngine


class FakeAsyncLLM:
    """Mimics vLLM AsyncLLM.encode: an async generator yielding a final
    PoolingRequestOutput-like object whose `.outputs.data` is the waveform."""

    def __init__(self):
        self.calls = []
        self.shutdown_called = False

    async def encode(self, prompt, pooling_params=None, request_id=None):
        self.calls.append(
            {"prompt": prompt, "pooling_params": pooling_params, "request_id": request_id}
        )

        class Out:
            class outputs:
                data = np.ones(1200, dtype=np.float32)

        yield Out()

    def shutdown(self):
        self.shutdown_called = True


def test_synthesize_chunk_returns_audio():
    eng = KokoroEngine(settings=None, _llm=FakeAsyncLLM())
    audio = asyncio.run(eng.synthesize_chunk([0, 5, 0], np.ones(256, np.float32), 1.0, "r1"))
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (1200,)


def test_is_ready_true_when_llm_injected():
    eng = KokoroEngine(settings=None, _llm=FakeAsyncLLM())
    assert eng.is_ready() is True


def test_is_ready_false_before_start_with_no_injected_llm():
    eng = KokoroEngine(settings=None)
    assert eng.is_ready() is False


def test_synthesize_chunk_builds_tokens_prompt_and_plugin_pooling_params():
    from vllm.pooling_params import PoolingParams

    fake = FakeAsyncLLM()
    eng = KokoroEngine(settings=None, _llm=fake)
    ref_s = np.arange(256, dtype=np.float32)
    asyncio.run(eng.synthesize_chunk([1, 2, 3], ref_s, 1.0, "req-42"))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["request_id"] == "req-42"
    # TokensPrompt is a TypedDict (plain dict at runtime, no isinstance check).
    assert isinstance(call["prompt"], dict)
    assert call["prompt"]["prompt_token_ids"] == [1, 2, 3]
    np.testing.assert_array_equal(call["prompt"]["multi_modal_data"]["voice"], ref_s)
    assert isinstance(call["pooling_params"], PoolingParams)
    assert call["pooling_params"].task == "plugin"


def test_synthesize_chunk_threads_speed_into_mm_processor_kwargs():
    fake = FakeAsyncLLM()
    eng = KokoroEngine(settings=None, _llm=fake)
    ref_s = np.ones(256, dtype=np.float32)
    asyncio.run(eng.synthesize_chunk([1, 2, 3], ref_s, 1.35, "req-1"))

    prompt = fake.calls[0]["prompt"]
    assert prompt.get("mm_processor_kwargs") == {"speed": 1.35}


def test_close_shuts_down_llm_and_clears_ready():
    fake = FakeAsyncLLM()
    eng = KokoroEngine(settings=None, _llm=fake)
    asyncio.run(eng.close())
    assert fake.shutdown_called is True
    assert eng.is_ready() is False
