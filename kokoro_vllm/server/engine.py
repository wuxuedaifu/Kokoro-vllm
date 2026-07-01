"""Async engine lifecycle wrapper around vLLM's `AsyncLLM`.

`KokoroEngine` is the seam between the FastAPI request layer (Tasks 15/16)
and the vLLM V1 pooling engine that actually runs Kokoro synthesis. It owns:

* `start()` — constructs the real `vllm.v1.engine.async_llm.AsyncLLM` using
  the exact engine kwargs proven at Task 12 (GPU parity gate, correlation =
  1.0 vs. the reference `kokoro.KModel`), after registering the Kokoro
  architecture/config via `kokoro_vllm.model.register.register()`.
* `synthesize_chunk()` — submits one `AsyncLLM.encode` request per phoneme
  chunk and returns the resulting waveform as a 1-D `float32` numpy array.
* `close()` — shuts the engine down.
* `is_ready()` — reports whether the engine is constructed and usable.

For unit testing (no GPU), a fake engine object satisfying the same
`.encode(prompt, pooling_params=..., request_id=...)` async-generator
protocol can be injected via the `_llm` constructor argument, bypassing
`start()` entirely.

IMPORTANT — real boot is NOT exercised by this module's tests. The
`start()` code path that builds a real `AsyncLLM` requires a GPU and is
validated at Task 16/17 (server boot / integration tests), not here.
"""

from __future__ import annotations

import numpy as np

from kokoro_vllm.config import ServerSettings


class KokoroEngine:
    """Lifecycle + request wrapper around vLLM's `AsyncLLM` for Kokoro TTS."""

    def __init__(self, settings: ServerSettings | None, _llm=None):
        self.settings = settings
        self._llm = _llm
        # An injected fake/real engine is immediately usable; otherwise the
        # engine only becomes ready once `start()` has run.
        self._ready = _llm is not None

    async def start(self) -> None:
        """Boot the real vLLM AsyncLLM engine.

        Uses the exact kwargs that achieved parity=1.0 against the reference
        KModel at Task 12 (see .superpowers/sdd/task-12-report.md §4),
        translated from the sync `LLM(...)` constructor into
        `AsyncEngineArgs` for `AsyncLLM.from_engine_args`. Not exercised by
        this task's tests (requires a GPU) — validated at Task 16/17.
        """
        if self._llm is None:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            from kokoro_vllm.model.register import register

            # Must happen before AsyncLLM construction so the "kokoro"
            # architecture/config are registered in-process.
            register()

            args = AsyncEngineArgs(
                model=self.settings.model_dir,
                runner="pooling",
                enforce_eager=True,
                # Pinned to 1, NOT `self.settings.max_num_seqs` (default 64).
                # Task 11/12 established that the current model only
                # supports single-request batching: with >1 request in a
                # forward pass, the `.shared` multimodal "voice" kwargs
                # collapse (all but the last request's ref_s is lost) and
                # `_per_request_token_spans` raises `NotImplementedError`
                # for num_requests>1 *inside* the engine-core loop, which
                # can crash the whole engine rather than cleanly rejecting
                # one request. `ServerSettings.max_num_seqs` is kept as a
                # field for the future `.shared`->`.batched` multimodal
                # upgrade (the throughput follow-up) but is intentionally
                # not used here until that lands.
                max_num_seqs=1,
                max_model_len=512,
                skip_tokenizer_init=True,
                dtype="float32",
                enable_flashinfer_autotune=False,
                trust_remote_code=True,
            )
            self._llm = AsyncLLM.from_engine_args(args)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def synthesize_chunk(
        self,
        input_ids: list[int],
        ref_s: np.ndarray,
        speed: float,
        request_id: str,
    ) -> np.ndarray:
        """Submit one `AsyncLLM.encode` request and return the waveform.

        Prompt/params match Task 12's proven config: a `TokensPrompt` with
        the phoneme ids + `voice` multimodal data, and
        `PoolingParams(task="plugin")` (the "plugin" pooling task is
        Kokoro's escape hatch for non-embedding/non-classification pooling
        output, per docs/superpowers/plans/vllm-interface-notes.md §1/§4).

        `speed` is threaded via `TokensPrompt.mm_processor_kwargs` per Task
        9's design; Task 12's parity run only exercised the default
        speed=1.0 path, so Task 16/17 must confirm this is how the model's
        multimodal processor actually consumes `speed` end-to-end.

        Note: `AsyncLLM.encode` (unlike the sync `LLM.encode`) takes no
        `pooling_task` kwarg — the task is set solely on `PoolingParams`
        (confirmed by inspecting the installed vllm 0.24.0 signature; see
        docs/superpowers/plans/vllm-interface-notes.md §4).
        """
        from vllm.inputs.llm import TokensPrompt
        from vllm.pooling_params import PoolingParams

        prompt = TokensPrompt(
            prompt_token_ids=list(input_ids),
            multi_modal_data={"voice": np.asarray(ref_s, dtype=np.float32)},
            mm_processor_kwargs={"speed": speed},
        )

        final = None
        async for out in self._llm.encode(
            prompt,
            pooling_params=PoolingParams(task="plugin"),
            request_id=request_id,
        ):
            final = out

        if final is None:
            raise RuntimeError(
                f"AsyncLLM.encode produced no output for request_id={request_id!r}"
            )
        return np.asarray(final.outputs.data, dtype=np.float32).reshape(-1)

    async def close(self) -> None:
        if self._llm is not None and hasattr(self._llm, "shutdown"):
            self._llm.shutdown()
        self._ready = False
