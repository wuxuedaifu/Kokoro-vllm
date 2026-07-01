"""Multimodal plumbing for Kokoro's `ref_s` voice-conditioning vector.

`ref_s` (float32[256]) is per-request GLOBAL conditioning for the TTS
forward pass -- analogous to XTTS's `cond_latents`. It is NOT expanded into
per-token placeholder embeddings the way image/video tokens are: the model
reads it directly inside `forward()` (see Task 11's
`KokoroForConditionalGeneration`). Consequently the multimodal processor
below never inserts prompt placeholders for the "voice" modality; its whole
job is to validate/normalize the incoming tensor and thread it (plus the
scalar `speed` value passed via `mm_processor_kwargs`) through to
`mm_kwargs` so it reaches the model's `forward()`/pooler unchanged.

`normalize_ref_s` is the single source of truth for `ref_s` validation --
it is used both directly (e.g. by callers building requests) and inside
`KokoroMultiModalDataParser` when parsing the raw `multi_modal_data["voice"]`
value.

The `KokoroMultiModalProcessor` / `KokoroProcessingInfo` /
`KokoroDummyInputsBuilder` triple below is registered by Task 11 via:

    @MULTIMODAL_REGISTRY.register_processor(
        KokoroMultiModalProcessor,
        info=KokoroProcessingInfo,
        dummy_inputs=KokoroDummyInputsBuilder,
    )
    class KokoroForConditionalGeneration(..., SupportsMultiModal, VllmModelForPooling):
        ...

See docs/superpowers/plans/vllm-interface-notes.md (section 3) for the
verified vllm 0.24.0 `register_processor` signature this triple targets, and
`vllm/model_executor/models/terratorch.py` for the closest in-tree precedent
for a raw-tensor, no-placeholder modality (this file follows that pattern).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)

VOICE_MODALITY = "voice"
REF_S_DIM = 256


def normalize_ref_s(ref_s) -> torch.Tensor:
    """Validate + normalize a `ref_s` voice-conditioning vector.

    Accepts a list, numpy array, or torch tensor of 256 floats (any shape
    that flattens to 256 elements) and returns a contiguous `float32[256]`
    tensor. Raises `ValueError` if the flattened element count isn't 256.

    This is the single source of truth for `ref_s` validation: it is used
    both by callers directly and inside `KokoroMultiModalDataParser` below.
    """
    if isinstance(ref_s, torch.Tensor):
        t = ref_s.detach().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(ref_s, dtype=np.float32))
    t = t.reshape(-1)
    if t.shape != (REF_S_DIM,):
        raise ValueError(f"ref_s must have {REF_S_DIM} elements, got {tuple(t.shape)}")
    return t.contiguous()


class _KokoroVoiceItems(ModalityDataItems[torch.Tensor, torch.Tensor]):
    """A single `ref_s` item for the "voice" modality.

    Kokoro only ever supports one voice-conditioning vector per request, so
    this always wraps exactly one item (`get_count() == 1`).
    """

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data, VOICE_MODALITY)

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(f"voice modality only has 1 item, got index {index}")
        return self.data

    def get_processor_data(self) -> Mapping[str, object]:
        # ref_s needs no HF-processor preprocessing; it's passed through as-is.
        return {}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {VOICE_MODALITY: self.data}


class KokoroMultiModalDataParser(MultiModalDataParser):
    """Parses `multi_modal_data["voice"]` into a `_KokoroVoiceItems`.

    The base `MultiModalDataParser` only knows about "audio"/"image"/"video"/
    "vision_chunk" -- a custom modality name like "voice" must add its own
    subparser or `parse_mm_data` raises `ValueError: Unsupported modality`.
    """

    def _parse_voice_data(
        self, data: ModalityData
    ) -> ModalityDataItems | None:
        if data is None:
            return None
        return _KokoroVoiceItems(normalize_ref_s(data))

    def _get_subparsers(self):
        return {**super()._get_subparsers(), VOICE_MODALITY: self._parse_voice_data}


class KokoroProcessingInfo(BaseProcessingInfo):
    """Declares the "voice" modality's limits for Kokoro."""

    def get_data_parser(self) -> MultiModalDataParser:
        return KokoroMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Exactly one ref_s vector per request.
        return {VOICE_MODALITY: 1}


class KokoroDummyInputsBuilder(BaseDummyInputsBuilder[KokoroProcessingInfo]):
    """Builds dummy `ref_s` data for engine profiling/warmup."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # No text placeholders are inserted for the "voice" modality; the
        # phoneme-token prompt is provided separately by the caller.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        if mm_counts.get(VOICE_MODALITY, 0) <= 0:
            return {}
        return {VOICE_MODALITY: normalize_ref_s(torch.zeros(REF_S_DIM))}


class KokoroMultiModalProcessor(BaseMultiModalProcessor[KokoroProcessingInfo]):
    """Threads `ref_s` (+ `speed` via `mm_processor_kwargs`) into `mm_kwargs`.

    `ref_s` is global conditioning, not a placeholder-token modality, so
    (like `TerratorchMultiModalProcessor`, the closest in-tree precedent for
    a raw-tensor modality) this processor bypasses the HF-processor-based
    pipeline entirely and overrides `apply()` directly: there is no
    `transformers.ProcessorMixin` for Kokoro voice conditioning, and no
    prompt placeholders need to be inserted or located.
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields: dict[str, MultiModalFieldConfig] = {}
        if VOICE_MODALITY in hf_inputs:
            fields[VOICE_MODALITY] = MultiModalFieldConfig.shared(
                VOICE_MODALITY, batch_size=1
            )
        if "speed" in hf_inputs:
            # `speed` shares the "voice" modality's item count (it always
            # travels alongside the ref_s vector for a given request).
            fields["speed"] = MultiModalFieldConfig.shared(VOICE_MODALITY, batch_size=1)
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # ref_s is global conditioning: no placeholder tokens are inserted
        # into (or located in) the prompt.
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        mm_items = inputs.mm_data_items
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs

        prompt = inputs.prompt
        if isinstance(prompt, list):
            prompt_token_ids = prompt
        else:
            prompt_token_ids = self.info.get_tokenizer().encode(prompt)

        with timing_ctx.record("apply_hf_processor"):
            _, passthrough_data = self._get_hf_mm_data(mm_items)
            passthrough_data = dict(passthrough_data)

            speed = hf_processor_mm_kwargs.get("speed")
            if speed is not None:
                passthrough_data["speed"] = torch.tensor(
                    float(speed), dtype=torch.float32
                )

            mm_processed_data = BatchFeature(
                {
                    k: torch.as_tensor(v).unsqueeze(0)
                    for k, v in passthrough_data.items()
                },
                tensor_type="pt",
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(mm_processed_data, hf_processor_mm_kwargs),
        )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        mm_placeholders = {VOICE_MODALITY: [PlaceholderRange(offset=0, length=0)]}

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )
