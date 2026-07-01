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
_MODEL_TYPE = "kokoro"


def register() -> None:
    """Register Kokoro with vLLM's model registry and transformers' config map.

    Two registrations are needed:

    * ``ModelRegistry.register_model`` makes ``KokoroForConditionalGeneration``
      loadable by architecture name (used by the model loader in the engine).
    * ``AutoConfig.register`` teaches transformers about ``model_type="kokoro"``
      so vLLM's ``ModelConfig`` can parse the converted ``config.json`` at
      ``LLM(...)`` construction time. Without this, ModelConfig raises
      "Transformers does not recognize this architecture" before any plugin /
      model code runs.
    """
    from transformers import AutoConfig

    from kokoro_vllm.model.hf_config import KokoroConfig
    from kokoro_vllm.model.kokoro_vllm_model import KokoroForConditionalGeneration

    try:
        AutoConfig.register(_MODEL_TYPE, KokoroConfig)
    except ValueError:
        # Already registered (idempotent across processes / repeated calls).
        pass

    if _ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(_ARCH, KokoroForConditionalGeneration)
