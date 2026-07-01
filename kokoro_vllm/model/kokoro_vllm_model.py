"""`KokoroForConditionalGeneration` -- the vLLM V1 pooling model for Kokoro TTS.

This module ties together every prior piece of the "Kokoro on vLLM V1" plan:

- the real `kokoro.KModel` graph (`build_kmodel`, Task 8),
- the raw->`kmodel.` state-dict key remap (`remap_param_name`, Task 8),
- the `ref_s` "voice" multimodal processor triple (Task 9),
- the identity waveform pooler (`KokoroWaveformPooler`, Task 10),
- the `KokoroConfig` HF config (Task 7).

Design decisions (verified against the installed vllm 0.24.0 source, see
`docs/superpowers/plans/vllm-interface-notes.md` and the inline references
below):

* **Raw-input multimodal pooling model.** Kokoro's `ref_s` voice vector is
  *global* conditioning (no placeholder-token expansion), and the phoneme
  `input_ids` must reach `forward` verbatim. The closest in-tree precedent is
  `vllm/model_executor/models/terratorch.py` (`Terratorch`): an
  attention-free, raw-input-only pooling model whose `forward(**kwargs)`
  consumes the raw multimodal tensors and whose pooler is an identity
  pass-through. We mirror that pattern exactly, adding two flags Terratorch
  does not need:
    - `requires_raw_input_tokens = True` so the model runner hands the real
      phoneme `input_ids` to `forward` (see
      `GPUModelRunner._prepare_mm_inputs`: it returns `self.input_ids.gpu`
      only when this flag is set, otherwise `None`). Kokoro *needs* the
      tokens, unlike Terratorch which only uses image kwargs.
    - `supports_multimodal_raw_input_only = True` so the runner routes the
      "voice"/"speed" kwargs straight into `forward(**kwargs)` via
      `GPUModelRunner._extract_mm_kwargs` instead of running a multimodal
      encoder.

* **Attention-free.** Kokoro runs its own internal attention inside `KModel`;
  vLLM manages no KV cache for it. `IsAttentionFree` + `@attn_type(
  "attention_free")` mirror Terratorch and keep vLLM from allocating attention
  backends / KV caches.

* **Pooler contract (Task 10).** `forward` returns a `list[torch.Tensor]`,
  one 1-D waveform per request; `KokoroWaveformPooler.forward` passes that
  list through unchanged. The pooling runner
  (`GPUModelRunner._pool`) then maps the list element-wise to per-request
  `PoolingRequestOutput`s.

KNOWN LIMITATION (documented for Task 12): the "voice" field is declared with
`MultiModalFieldConfig.shared` (Task 9). For raw-input-only models the runner
combines per-request mm kwargs with a plain `dict.update()` over the groups
produced by `group_and_batch_mm_kwargs`, and *shared* fields with distinct
per-request values land in distinct groups -- so when more than one request is
batched into a single `forward`, only the **last** request's `voice`/`speed`
survives in `kwargs`. Consequently this model is only guaranteed correct with
**one request per forward** (e.g. `max_num_seqs=1`, which is the natural mode
for the Task 13 per-request `AsyncLLM.encode` path). Multi-request batching
would require switching "voice" to `MultiModalFieldConfig.batched` *and*
recovering per-request phoneme lengths (see `_per_request_token_spans`).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    IsAttentionFree,
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    attn_type,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from kokoro_vllm.model.kmodel_access import build_kmodel
from kokoro_vllm.model.mm_processor import (
    REF_S_DIM,
    VOICE_MODALITY,
    KokoroDummyInputsBuilder,
    KokoroMultiModalProcessor,
    KokoroProcessingInfo,
)
from kokoro_vllm.model.pooler import KokoroWaveformPooler
from kokoro_vllm.model.weights import remap_param_name

# Kokoro's iSTFTNet decoder injects *unseeded* randomness into every synthesis:
# a random initial phase (`torch.rand`) and Gaussian excitation noise
# (`torch.randn_like`) in `kokoro.istftnet`'s `SineGen`/source module. This
# makes KModel.forward_with_tokens non-deterministic run-to-run (~0.99
# self-correlation on the raw waveform) even with identical weights and inputs.
# We seed the RNG to a fixed value immediately before synthesis so that a given
# (phonemes, ref_s, speed) yields reproducible audio, and so the GPU parity gate
# can compare the vLLM engine output against an identically-seeded reference
# KModel (see tests/model/test_parity_gpu.py). 0 matches vLLM's default engine
# seed.
KOKORO_SYNTHESIS_SEED = 0


def _per_request_token_spans(
    input_ids: torch.Tensor, num_requests: int
) -> list[torch.Tensor]:
    """Split a flattened 1-D `input_ids` batch into per-request token runs.

    vLLM hands `forward` a single 1-D `input_ids` tensor with every scheduled
    request's tokens concatenated. For the guaranteed-correct single-request
    case (see module docstring) there is exactly one span covering all tokens.

    For `num_requests > 1` there is no per-request phoneme-length signal
    available to an *attention-free* pooling model inside `forward`
    (`PoolingMetadata` is only handed to the pooler, and attention-free models
    carry no `query_start_loc` in the forward context). Rather than silently
    mis-split, we fail loudly and point Task 12 at the fix. See the KNOWN
    LIMITATION note in the module docstring.
    """
    if input_ids.dim() != 1:
        input_ids = input_ids.reshape(-1)

    if num_requests <= 1:
        return [input_ids]

    raise NotImplementedError(
        "KokoroForConditionalGeneration.forward received "
        f"{num_requests} requests in one batch, but per-request phoneme "
        "lengths are not recoverable for an attention-free pooling model, "
        "and the 'voice' kwargs for all but the last request are lost to "
        "the raw-input-only dict.update() combine (see module docstring). "
        "Run with max_num_seqs=1, or switch the 'voice' multimodal field to "
        "MultiModalFieldConfig.batched and thread per-request token lengths "
        "into forward before enabling multi-request batching."
    )


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    KokoroMultiModalProcessor,
    info=KokoroProcessingInfo,
    dummy_inputs=KokoroDummyInputsBuilder,
)
class KokoroForConditionalGeneration(
    nn.Module, IsAttentionFree, SupportsMultiModal, VllmModelForPooling
):
    """vLLM V1 pooling model wrapping the real `kokoro.KModel`."""

    is_pooling_model = True
    # ref_s is fed as a raw tensor straight to forward(); no encoder runs.
    supports_multimodal_raw_input_only = True
    # Kokoro needs the raw phoneme token ids in forward(), not input embeds.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # "voice" is global conditioning, not a placeholder-token modality.
        if modality == VOICE_MODALITY:
            return None
        raise ValueError(f"Unsupported modality for Kokoro: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        device_config = vllm_config.device_config
        device = getattr(device_config, "device", None)

        self.config = config
        # Real KModel submodule graph (bert, bert_encoder, predictor,
        # text_encoder, decoder) with random weights; real weights arrive via
        # load_weights().
        self.kmodel = build_kmodel(config, str(device) if device is not None else "cpu")

        # Identity waveform pooler: forward() already returns per-request
        # audio tensors, which this pooler passes through unchanged.
        self.pooler = KokoroWaveformPooler()

        # Width of the dummy input-embeds tensor the runner copies from
        # embed_input_ids(); read from the config so the copy shapes match
        # regardless of how the engine resolves the hidden size.
        try:
            self._inputs_embeds_size = int(
                vllm_config.model_config.get_inputs_embeds_size()
            )
        except Exception:  # pragma: no cover - defensive; validated at Task 12
            self._inputs_embeds_size = int(getattr(config, "hidden_dim", 0))

    # ------------------------------------------------------------------
    # SupportsMultiModal hooks
    # ------------------------------------------------------------------
    def get_language_model(self) -> nn.Module:
        # Kokoro has no separable "language model" sub-stack; the whole KModel
        # is driven from forward(). Return self so callers that probe for a
        # language model get a valid module.
        return self

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # ref_s is global conditioning consumed directly in forward(), never
        # merged into token embeddings, so there are no multimodal embeddings.
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # No token embeddings are actually used (forward() reads raw
        # input_ids). We still return a correctly-shaped zero tensor because
        # the model runner copies this into its inputs_embeds buffer of width
        # `inputs_embeds_size` before calling forward().
        return input_ids.new_zeros(
            (input_ids.shape[0], self._inputs_embeds_size), dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError(
                "KokoroForConditionalGeneration.forward requires raw input_ids "
                "(set requires_raw_input_tokens=True)."
            )

        voice = kwargs.get(VOICE_MODALITY)
        if voice is None:
            raise ValueError(
                "Missing 'voice' (ref_s) multimodal kwarg in Kokoro forward()."
            )
        ref_s = torch.as_tensor(voice)
        # Normalize to (num_requests, REF_S_DIM). The mm processor stores each
        # request's ref_s with a leading batch dim (shape (1, 256)); a bare
        # (256,) vector is also accepted.
        ref_s = ref_s.reshape(-1, REF_S_DIM)
        num_requests = ref_s.shape[0]

        speed = self._extract_speed(kwargs.get("speed"))

        device = next(self.kmodel.parameters()).device

        spans = _per_request_token_spans(input_ids, num_requests)

        # Make the stochastic decoder deterministic (see KOKORO_SYNTHESIS_SEED).
        torch.manual_seed(KOKORO_SYNTHESIS_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(KOKORO_SYNTHESIS_SEED)

        audios: list[torch.Tensor] = []
        for req_ids, req_ref_s in zip(spans, ref_s):
            ids = req_ids.to(device=device, dtype=torch.long).reshape(1, -1)
            rs = req_ref_s.to(device=device, dtype=torch.float32).reshape(1, REF_S_DIM)
            audio, _pred_dur = self.kmodel.forward_with_tokens(ids, rs, speed)
            audios.append(audio.reshape(-1))

        # Return a 2-D tensor of shape (num_requests, audio_len). Crucially,
        # dim 0 is the *request* axis, not the audio-sample axis. vLLM's pooling
        # runner slices the model output as `hidden_states[:num_scheduled_tokens]`
        # in `_pool` (GPUModelRunner) and reads `hidden_states.shape[0]` during
        # the pooler warmup (`_dummy_pooler_run_task`). If dim 0 were the audio
        # length, the `[:num_scheduled_tokens]` slice would truncate the waveform
        # to the number of phoneme tokens. With dim 0 == num_requests (== 1 under
        # the guaranteed single-request mode) the slice is a no-op and warmup
        # sees a well-formed tensor. `KokoroWaveformPooler` then splits this back
        # into one 1-D waveform per request (row).
        return torch.stack(audios, dim=0)

    @staticmethod
    def _extract_speed(speed) -> float:
        if speed is None:
            return 1.0
        if isinstance(speed, torch.Tensor):
            return float(speed.reshape(-1)[0].item())
        try:
            return float(speed)
        except (TypeError, ValueError):
            return 1.0

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load raw KModel weights, remapping each key under `kmodel.`.

        `remap_param_name` prefixes every incoming raw KModel key (e.g.
        `bert.embeddings...`, `decoder...`) with `kmodel.` so it lines up with
        this module's parameter/buffer tree. We copy into both parameters and
        buffers (KModel carries non-persistent buffers as well).
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()

        for raw_name, weight in weights:
            name = remap_param_name(raw_name)
            target = params.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                # Unknown/extra key; skip (Task 12 verifies the full mapping
                # against real converted weights).
                continue
            with torch.no_grad():
                target.copy_(weight.to(dtype=target.dtype, device=target.device))
            loaded.add(name)

        # The real kokoro-v1_0.pth checkpoint deliberately omits the affine
        # parameters of the decoder's AdaIN InstanceNorm1d layers
        # (`decoder.*.norm.{weight,bias}`). `AdaIN1d` sets `affine=True` only to
        # work around an old torch.onnx.export bug; those affine params are
        # never trained and stay at their default identity init (weight=1,
        # bias=0), with the real per-channel modulation coming from `AdaIN1d.fc`.
        # The reference `KModel` loads the same checkpoint with strict=False and
        # therefore also leaves these at 1/0, so parity is preserved. We must
        # still satisfy vLLM's boot-time coverage check
        # (default_loader.track_weights_loading raises if any named_parameter is
        # unloaded), so explicitly reset these to their identity init and mark
        # them loaded.
        with torch.no_grad():
            for name, target in params.items():
                if name in loaded:
                    continue
                if name.endswith(".norm.weight"):
                    target.fill_(1.0)
                    loaded.add(name)
                elif name.endswith(".norm.bias"):
                    target.fill_(0.0)
                    loaded.add(name)

        return loaded
