import torch

from anima_slam_gs3lam.semantic.decoder import SemanticDecoder


def test_decoder_outputs_logits():
    decoder = SemanticDecoder(in_channels=16, out_channels=256)
    feature_map = torch.randn(16, 12, 20)
    logits = decoder(feature_map)
    assert logits.shape == (256, 12, 20)


def test_decoder_rejects_wrong_channel_count():
    decoder = SemanticDecoder(in_channels=16, out_channels=256)
    feature_map = torch.randn(8, 6, 6)
    try:
        decoder(feature_map)
    except ValueError as exc:
        assert "channel dimension" in str(exc)
    else:
        raise AssertionError("decoder should reject mismatched channels")
