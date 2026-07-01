import torch

from kokoro_vllm.model.pooler import KokoroWaveformPooler, _slice_per_request


def test_slice_per_request():
    flat = torch.arange(10, dtype=torch.float32)
    out = _slice_per_request(flat, [3, 7])
    assert len(out) == 2
    assert torch.equal(out[0], torch.arange(0, 3, dtype=torch.float32))
    assert torch.equal(out[1], torch.arange(3, 10, dtype=torch.float32))


def test_slice_per_request_three_way_split():
    flat = torch.arange(6, dtype=torch.float32)
    out = _slice_per_request(flat, [1, 2, 3])
    assert len(out) == 3
    assert torch.equal(out[0], torch.tensor([0.0]))
    assert torch.equal(out[1], torch.tensor([1.0, 2.0]))
    assert torch.equal(out[2], torch.tensor([3.0, 4.0, 5.0]))


def test_slice_per_request_empty_lengths():
    flat = torch.empty(0, dtype=torch.float32)
    assert _slice_per_request(flat, []) == []


def test_kokoro_waveform_pooler_has_no_abstract_methods():
    assert KokoroWaveformPooler.__abstractmethods__ == frozenset()


def test_kokoro_waveform_pooler_supports_plugin_task():
    pooler = KokoroWaveformPooler()
    assert pooler.get_supported_tasks() == {"plugin"}


def test_kokoro_waveform_pooler_forward_passthrough_for_list_input():
    pooler = KokoroWaveformPooler()
    per_request_audio = [torch.arange(3, dtype=torch.float32), torch.arange(7, dtype=torch.float32)]

    # pooling_metadata is unused on the list-passthrough path; a sentinel is
    # enough to prove forward() doesn't touch it.
    out = pooler.forward(per_request_audio, pooling_metadata=object())

    assert out is per_request_audio
