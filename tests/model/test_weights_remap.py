from kokoro_vllm.model.weights import remap_param_name


def test_prefixes_kmodel():
    assert remap_param_name("bert.encoder.layer.0.weight") == "kmodel.bert.encoder.layer.0.weight"


def test_predictor_and_decoder():
    assert remap_param_name("predictor.lstm.weight_ih_l0") == "kmodel.predictor.lstm.weight_ih_l0"
    assert remap_param_name("decoder.generator.conv.weight") == "kmodel.decoder.generator.conv.weight"


def test_idempotent_when_already_prefixed():
    assert remap_param_name("kmodel.bert.x") == "kmodel.bert.x"
