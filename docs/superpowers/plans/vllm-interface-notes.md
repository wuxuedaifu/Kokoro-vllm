# vLLM V1 pooling / multimodal interface notes

Captured against the exact package installed via `uv pip install -e ".[vllm]"` in this
repo's venv on 2026-06-30/07-01.

- **Resolved `vllm` version:** `0.24.0`
- **Resolved `torch` version:** `2.11.0` (CUDA build, pulled in as a transitive dep of vllm)
- **Python:** 3.12 (repo venv, `.venv/lib/python3.12/site-packages/vllm`)
- Verified with:
  ```
  uv run python -c "import vllm; print(vllm.__version__)"
  # -> 0.24.0
  ```

All four import paths given in the task brief (`vllm.model_executor.models.interfaces.SupportsMultiModal`,
`vllm.model_executor.models.interfaces_base.VllmModelForPooling`,
`vllm.model_executor.layers.pooler.Pooler`, `vllm.v1.engine.async_llm.AsyncLLM`) **resolved
without error** on 0.24.0 — no corrected top-level import paths were needed. However, several
*method names* referenced only loosely/by-example in the brief have been renamed relative to
older vLLM releases; these are called out explicitly below since later tasks will code directly
against them.

---

## 1. `Pooler` required abstract methods + signatures

Import: `from vllm.model_executor.layers.pooler import Pooler`

`Pooler` is an `nn.Module` + `ABC`. Its abstract method set on 0.24.0:

```
Pooler.__abstractmethods__ == frozenset({'forward', 'get_supported_tasks'})
```

Exact signatures (via `inspect.signature`):

```python
Pooler.forward(self, hidden_states: torch.Tensor, pooling_metadata: vllm.v1.pool.metadata.PoolingMetadata) -> torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]

Pooler.get_supported_tasks(self) -> collections.abc.Set[typing.Literal['embed', 'classify', 'token_embed', 'token_classify', 'plugin', 'embed&token_classify']]
```

`get_pooling_updates` is **not** abstract (it has a base implementation) but is part of the
public contract subclasses are expected to override when they change activation/normalization
behavior:

```python
Pooler.get_pooling_updates(self, task: Literal['embed', 'classify', 'token_embed', 'token_classify', 'plugin', 'embed&token_classify']) -> vllm.model_executor.layers.pooler.common.PoolingParamsUpdate
```

Note the `PoolingTask` literal (`from vllm.pooling_params import PoolingTask`) includes a
`'plugin'` task in addition to `'embed'` / `'classify'` / `'token_embed'` / `'token_classify'` /
`'embed&token_classify'`. `'plugin'` is the escape hatch for pooling outputs that are neither an
embedding nor a classification score (i.e. it is the task Kokoro's audio-tensor pooler should
declare via `get_supported_tasks`). When `PoolingParams.task == "plugin"`, `PoolingParams.verify()`
skips its normal parameter validation (`vllm/pooling_params.py`, `PoolingParams.verify`):

```python
def verify(self, model_config: ModelConfig) -> None:
    # plugin task uses io_processor.parse_request to verify inputs,
    # skipping PoolingParams verify
    if self.task == "plugin":
        if self.skip_reading_prefix_cache is None:
            self.skip_reading_prefix_cache = True
        return
    ...
```

There is also a ready-made dispatch helper, `DispatchPooler`
(`from vllm.model_executor.layers.pooler import DispatchPooler`), that maps a pooling task name
to a concrete `Pooler` instance:

```python
DispatchPooler.__init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None
DispatchPooler.for_embedding(pooler_config: vllm.config.pooler.PoolerConfig)   # classmethod, "embed" task convenience ctor
```

For Kokoro (task 10/11), the plan should be: implement a custom `Pooler` subclass whose
`get_supported_tasks()` returns `{"plugin"}` and whose `forward(hidden_states, pooling_metadata)`
returns the generated audio tensor(s) (one `torch.Tensor` per request, ragged/list-of-tensors is a
legal `forward` return type per the signature above), then wrap it in a `DispatchPooler({"plugin": KokoroPooler(...)})`
or assign it directly as `self.pooler`.

---

## 2. How a model declares itself a pooling model in 0.24.0

Base class / mixin: **`vllm.model_executor.models.interfaces_base.VllmModelForPooling`**
(`from vllm.model_executor.models.interfaces_base import VllmModelForPooling`).

It is a `@runtime_checkable` `Protocol`, not an `ABC` — there is no metaclass enforcement, but
`vllm.model_executor.models.interfaces_base.is_pooling_model(obj_or_cls)` is the runtime check
vLLM's model loader uses (`getattr(model, "is_pooling_model", False)` under the hood):

```python
def is_pooling_model(
    model: type[object] | object,
) -> TypeIs[type[VllmModelForPooling]] | TypeIs[VllmModelForPooling]:
    if not is_vllm_model(model):
        return False
    return getattr(model, "is_pooling_model", False)
```

The protocol's class-level contract (from `interfaces_base.py`):

```python
@runtime_checkable
class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    is_pooling_model: ClassVar[Literal[True]] = True
    default_seq_pooling_type: ClassVar[SequencePoolingType] = "LAST"
    default_tok_pooling_type: ClassVar[TokenPoolingType] = "ALL"
    attn_type: ClassVar[AttnTypeStr] = "decoder"
    score_type: ClassVar[ScoreType] = "bi-encoder"
    pooler: Pooler   # only called on TP rank 0
```

**Real-world usage pattern in this vLLM version** — a multimodal *and* pooling model
subclasses both interfaces directly, e.g. `vllm/model_executor/models/nemotron_vl.py`:

```python
@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[LlamaNemotronVLEmbedProcessingInfo],
    info=LlamaNemotronVLEmbedProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[LlamaNemotronVLEmbedProcessingInfo],
)
class LlamaNemotronVLForEmbedding(LlamaNemotronVLChatModel, VllmModelForPooling):
    is_pooling_model = True
    ...
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)
```

i.e. `LlamaNemotronVLChatModel` already implements `SupportsMultiModal`; the pooling subclass adds
`VllmModelForPooling` to the MRO, sets `is_pooling_model = True`, and assigns `self.pooler` in
`__init__`. **This is the exact pattern task 9/10 should follow for
`KokoroForConditionalGeneration(SomeBaseOrMixin, SupportsMultiModal, VllmModelForPooling)`.**

There is also an automatic "as-pooling-model" adapter for turning a `*ForCausalLM` into a pooling
model (`vllm/model_executor/models/adapters.py:_create_pooling_model_cls`), which dynamically
builds `class ModelForPooling(orig_cls, VllmModelForPooling): is_pooling_model = True` and calls
`self._init_pooler(vllm_config, prefix=prefix)` — but this path assumes a generation model being
repurposed; Kokoro should use the direct-subclass pattern above instead since it is pooling-only
from the start.

---

## 3. `SupportsMultiModal` required methods + multimodal registry API

Import: `from vllm.model_executor.models.interfaces import SupportsMultiModal`

`SupportsMultiModal` is also a `@runtime_checkable` `Protocol` (`__abstractmethods__` is an empty
`frozenset()` — nothing is enforced at class-creation time; conformance is checked structurally /
by convention). The methods a model is expected to implement (from `dir(SupportsMultiModal)` and
the source in `vllm/model_executor/models/interfaces.py`):

```python
SupportsMultiModal.get_language_model(self) -> vllm.model_executor.models.interfaces_base.VllmModel

@classmethod
SupportsMultiModal.get_placeholder_str(modality: str, i: int) -> str | None

SupportsMultiModal.embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...]

SupportsMultiModal.embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings: list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor
```

**IMPORTANT correction vs. the brief's example names**: the brief's snippet mentions
`get_multimodal_embeddings` and `get_input_embeddings` as example method names to look for. In
the *installed* 0.24.0 package these have been **renamed**:

- `get_multimodal_embeddings` → **`embed_multimodal`**
- `get_input_embeddings` → **`embed_input_ids`**

There is no `get_multimodal_embeddings` or `get_input_embeddings` attribute anywhere on
`SupportsMultiModal` in this version — later tasks (9/10) must implement `embed_multimodal` and
`embed_input_ids`, not the old names, or the model will silently fail to satisfy the protocol.

Other relevant class vars declared on the protocol:

```python
supports_multimodal: ClassVar[Literal[True]] = True
supports_multimodal_raw_input_only: ClassVar[bool] = False
supports_encoder_tp_data: ClassVar[bool] = False
requires_raw_input_tokens: ClassVar[bool] = False
```

Runtime structural check: `vllm.model_executor.models.interfaces.supports_multimodal(model)`
(`(model: type[object] | object) -> TypeIs[...]`).

### Multimodal registry / processor registration API

Import: `from vllm.multimodal import MULTIMODAL_REGISTRY` — singleton instance of
`vllm.multimodal.registry.MultiModalRegistry`.

Registration is done via a class decorator, **`MULTIMODAL_REGISTRY.register_processor`**:

```python
MultiModalRegistry.register_processor(
    processor: vllm.multimodal.registry.MultiModalProcessorFactory[_I],
    *,
    info: vllm.multimodal.registry.ProcessingInfoFactory[_I],
    dummy_inputs: vllm.multimodal.registry.DummyInputsBuilderFactory[_I],
)
```

Usage (from `nemotron_vl.py`, confirmed to still be the pattern in 0.24.0):

```python
@MULTIMODAL_REGISTRY.register_processor(
    SomeMultiModalProcessor[SomeProcessingInfo],
    info=SomeProcessingInfo,
    dummy_inputs=SomeDummyInputsBuilder[SomeProcessingInfo],
)
class MyModel(BaseModel, SupportsMultiModal, VllmModelForPooling):
    ...
```

This sets `model_cls._processor_factory = _ProcessorFactories(info=..., dummy_inputs=..., processor=...)`
on the decorated class (a `ClassVar[_ProcessorFactories]` declared on `SupportsMultiModal`, i.e.
`_processor_factory` — "Set internally by `MultiModalRegistry.register_processor`" per the
docstring in `interfaces.py`). For Kokoro's `ref_s` voice tensor, task 11 will need to write a
custom `BaseMultiModalProcessor`/`ProcessingInfo`/`BaseDummyInputsBuilder` triple and register it
with this exact decorator.

`@support_torch_compile` is a separate, orthogonal decorator (not part of the multimodal registry)
at **`vllm.compilation.decorators.support_torch_compile`**:

```python
support_torch_compile(
    cls: type[_T] | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int] | dict[int, str]] | None = None,
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[vllm.config.vllm.VllmConfig], bool] | None = None,
    is_encoder: bool = False,
) -> Callable[[type[_T]], type[_T]] | type[_T]
```

Model registration with vLLM's model registry (needed to make `KokoroForConditionalGeneration`
loadable by architecture name) is `vllm.ModelRegistry.register_model`:

```python
ModelRegistry.register_model(model_arch: str, model_cls: type[torch.nn.Module] | str) -> None
```

---

## 4. `.encode(...)` argument names + `PoolingRequestOutput` tensor field

### Prompt fields (`prompt_token_ids` + `multi_modal_data`)

Both `LLM.encode` and `AsyncLLM.encode` accept a `prompt` argument that can be a
`vllm.inputs.llm.TokensPrompt` (among other prompt types). `TokensPrompt` is a `TypedDict`
(`vllm/inputs/data.py`, re-exported at `vllm.inputs.llm.TokensPrompt`) with these relevant keys:

```python
TokensPrompt.__required_keys__ == frozenset({'prompt_token_ids'})
TokensPrompt.__optional_keys__ == frozenset({
    'token_type_ids', 'prompt', 'mm_processor_kwargs',
    'multi_modal_uuids', 'cache_salt', 'multi_modal_data',
})
TokensPrompt.__annotations__ == {
    'prompt_token_ids': list[int],
    'multi_modal_data': typing.NotRequired[Mapping[str, Any | list[Any | None] | None] | None],
    'mm_processor_kwargs': typing.NotRequired[dict[str, Any] | None],
    'multi_modal_uuids': typing.NotRequired[Mapping[str, Sequence[str | None] | str]],
    'cache_salt': typing.NotRequired[str],
    'prompt': typing.NotRequired[str],
    'token_type_ids': typing.NotRequired[list[int]],
}
```

So the caller builds:

```python
from vllm.inputs.llm import TokensPrompt
prompt = TokensPrompt(prompt_token_ids=[...], multi_modal_data={"ref_s": ref_s_tensor})
```

### `LLM.encode` signature (sync, offline batch API)

```python
LLM.encode(
    self,
    prompts: str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt | ExplicitEncoderDecoderPrompt | Sequence[...] | DataPrompt,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm_asyncio] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    pooling_task: Optional[Literal['embed', 'classify', 'token_embed', 'token_classify', 'plugin', 'embed&token_classify']] = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[vllm.outputs.PoolingRequestOutput]
```

### `AsyncLLM.encode` signature (the V1 async engine path task 13 calls)

Import: `from vllm.v1.engine.async_llm import AsyncLLM` (resolves without error on 0.24.0 —
brief's import path is correct as-is).

```python
AsyncLLM.encode(
    self,
    prompt: str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt | ExplicitEncoderDecoderPrompt | TokensInput | EmbedsInput | MultiModalInput | EncoderDecoderInput,
    pooling_params: PoolingParams,
    request_id: str,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    tokenization_kwargs: dict[str, Any] | None = None,
    reasoning_ended: bool | None = None,
) -> AsyncGenerator[vllm.outputs.PoolingRequestOutput, None]
```

So the call task 13 needs is, e.g.:

```python
from vllm.inputs.llm import TokensPrompt
from vllm.pooling_params import PoolingParams

async for output in async_llm.encode(
    prompt=TokensPrompt(prompt_token_ids=token_ids, multi_modal_data={"ref_s": ref_s}),
    pooling_params=PoolingParams(task="plugin"),
    request_id=request_id,
):
    ...
```

Note `AsyncLLM.encode` takes `prompt` (singular, positional-or-keyword) not `prompts`
(plural) — unlike the sync `LLM.encode`, which batches. It also does not take a `pooling_task`
kwarg; the task must instead be set on the `PoolingParams` object itself (`PoolingParams(task=...)`).

### `PoolingRequestOutput` tensor field

Import: `from vllm.outputs import PoolingRequestOutput, PoolingOutput`.

```python
PoolingRequestOutput.__init__(
    self, request_id: str, outputs: _O, prompt_token_ids: list[int],
    num_cached_tokens: int, finished: bool,
) -> None
```

`outputs` is a `PoolingOutput` (or subclass), whose sole field is `data`:

```python
@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request.

    Args:
        data: The extracted hidden states.
    """
    data: torch.Tensor
```

So the returned tensor (Kokoro's generated audio) is read as **`result.outputs.data`** — matching
the brief's example exactly. Confirmed by direct source inspection, not by memory.

---

## Summary of corrections vs. the brief's snippet

| Brief mentioned | Installed vllm 0.24.0 reality |
|---|---|
| `from vllm.model_executor.models.interfaces import SupportsMultiModal` | Correct, no change needed |
| `from vllm.model_executor.models.interfaces_base import VllmModelForPooling` | Correct, no change needed |
| `from vllm.model_executor.layers.pooler import Pooler` | Correct, no change needed |
| `from vllm.v1.engine.async_llm import AsyncLLM` | Correct, no change needed |
| example abstract methods `get_supported_tasks`, `get_pooling_updates`, `forward` | Actual abstract set is `{forward, get_supported_tasks}`; `get_pooling_updates` exists but is **not** abstract (has a default impl) |
| example MM methods `get_multimodal_embeddings`, `get_input_embeddings` | **Renamed** to `embed_multimodal` and `embed_input_ids` respectively — no methods with the old names exist |
| `.outputs.data` for the pooling tensor | Confirmed correct: `PoolingRequestOutput.outputs` is a `PoolingOutput` dataclass with field `data: torch.Tensor` |
