"""Waveform pooler for Kokoro.

Kokoro's model forward (Task 11) produces raw audio, not an embedding or a
classification score, so this pooler declares the vLLM 0.24.0 ``"plugin"``
pooling task (see ``vllm.pooling_params.PoolingParams.verify`` — the
"plugin" task skips the normal embed/classify parameter validation and is
the documented escape hatch for pooling outputs of this shape, e.g.
``vllm.model_executor.layers.pooler.special.IdentityPooler``).

Model -> pooler contract (Task 11 / Task 12):
    ``KokoroForConditionalGeneration.forward`` returns a **2-D
    ``torch.Tensor`` of shape ``(num_requests, audio_len)``** — the request
    axis MUST be dim 0, because vLLM's ``_pool`` slices the model output as
    ``hidden_states[:num_scheduled_tokens]`` and a flat 1-D waveform would
    be truncated to the token count. This pooler ``forward`` splits that
    tensor back into one ``torch.Tensor`` per request (one row each). A
    ``list[torch.Tensor]`` input is also accepted (identity passthrough);
    both are legal ``Pooler.forward`` return types per the real base-class
    signature (`torch.Tensor | list[torch.Tensor] | list[torch.Tensor |
    None]`). (Single-request batches only for now — see Task 11 notes on
    the ``.shared`` mm-kwargs batching limitation.)

    ``_slice_per_request`` is kept as a pure, independently-testable helper
    for a fallback path: if audio ever arrives as a single flattened
    tensor plus explicit per-request lengths, it splits that tensor back
    into per-request pieces. Note that vLLM's real ``PoolingMetadata``
    (``vllm.v1.pool.metadata.PoolingMetadata``) has no ``audio_lengths``
    field — nothing on it identifies how many audio samples belong to each
    request, only token-level info like ``prompt_lens``. So this fallback
    path requires the *caller* (i.e. the model) to supply lengths obtained
    from its own bookkeeping; there is no field to pull them from off
    ``pooling_metadata`` in the base pooling contract.
"""

from collections.abc import Set

import torch

from vllm.model_executor.layers.pooler import Pooler
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


def _slice_per_request(flat_audio: torch.Tensor, seq_lengths: list[int]) -> list[torch.Tensor]:
    """Split a concatenated 1-D audio tensor into per-request tensors."""
    out: list[torch.Tensor] = []
    start = 0
    for n in seq_lengths:
        out.append(flat_audio[start : start + n])
        start += n
    return out


class KokoroWaveformPooler(Pooler):
    """Passes each request's synthesized waveform through as its pooled output.

    No normalization or reduction is applied — Kokoro's decoder output *is*
    the desired result, unlike an embedding pooler that reduces hidden
    states to a fixed-size vector.
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        # Already-split path: the model forward handed us one tensor per
        # request directly.
        if isinstance(hidden_states, list):
            return hidden_states
        if isinstance(hidden_states, tuple):
            return list(hidden_states)

        # Expected path (see module docstring + the model's forward): the model
        # returns a 2-D tensor of shape (num_requests, audio_len) -- dim 0 is
        # the request axis, so each row is that request's 1-D waveform. Splitting
        # by row recovers the per-request audio tensors the pooling runner maps
        # element-wise to PoolingRequestOutputs.
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() == 1:
                # A single flattened waveform: treat it as one request's audio.
                return [hidden_states]
            return [hidden_states[i] for i in range(hidden_states.shape[0])]

        raise TypeError(
            "KokoroWaveformPooler.forward() received an unsupported "
            f"{type(hidden_states).__name__}; expected a per-request list or a "
            "(num_requests, audio_len) tensor from the model forward."
        )
