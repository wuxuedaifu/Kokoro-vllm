"""vLLM plugin registration for the Kokoro TTS pooling model.

`register()` is the entry point declared in `pyproject.toml` under
`[project.entry-points."vllm.general_plugins"]`. vLLM imports and calls every
such entry point during engine startup (via `load_general_plugins`), which
makes `KokoroForConditionalGeneration` loadable by architecture name.

Importing the model module also triggers the
`@MULTIMODAL_REGISTRY.register_processor(...)` decorator on the class, wiring
the "voice" (`ref_s`) multimodal processor/info/dummy-inputs triple.
"""

from vllm import ModelRegistry

_ARCH = "KokoroForConditionalGeneration"


def register() -> None:
    """Register `KokoroForConditionalGeneration` with vLLM's model registry."""
    from kokoro_vllm.model.kokoro_vllm_model import KokoroForConditionalGeneration

    if _ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(_ARCH, KokoroForConditionalGeneration)
