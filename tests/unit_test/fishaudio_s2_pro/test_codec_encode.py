# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Fish reference audio codes-only encoding."""

import math
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("dac")

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.rvq import (
    DownsampleResidualVectorQuantize,
    VQResult,
)


@pytest.fixture
def quantizer() -> DownsampleResidualVectorQuantize:
    torch.manual_seed(42)
    return DownsampleResidualVectorQuantize(
        input_dim=8,
        n_codebooks=3,
        codebook_dim=2,
        codebook_size=16,
        semantic_codebook_size=32,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        pre_module=nn.Conv1d(8, 8, 1),
        post_module=nn.Sequential(nn.Conv1d(8, 8, 1), nn.Dropout(0.5)),
    ).eval()


@pytest.fixture
def codec(quantizer: DownsampleResidualVectorQuantize) -> DAC:
    return DAC(
        encoder_dim=2,
        encoder_rates=[2],
        latent_dim=8,
        decoder_dim=4,
        decoder_rates=[2],
        encoder_transformer_layers=[0],
        decoder_transformer_layers=[0],
        quantizer=quantizer,
    ).eval()


@pytest.mark.parametrize("length", [1, 8, 9])
@pytest.mark.parametrize("n_quantizers", [None, 1, 3])
def test_codes_only_matches_full_quantizer(
    quantizer: DownsampleResidualVectorQuantize,
    length: int,
    n_quantizers: int | None,
) -> None:
    latent = torch.randn(2, 8, length)
    with torch.inference_mode():
        expected = quantizer(latent, n_quantizers=n_quantizers)
        codes = quantizer.encode_codes(latent, n_quantizers=n_quantizers)

    assert codes.dtype == torch.long
    assert codes.shape == (2, 1 + (n_quantizers or 3), math.ceil(length / 4))
    assert torch.equal(codes, expected.codes)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17])
@pytest.mark.parametrize("n_quantizers", [None, 1])
def test_codec_encode_preserves_padding_codes_and_lengths(
    codec: DAC, length: int, n_quantizers: int | None
) -> None:
    waveform = torch.randn(2, length)
    audio_lengths = torch.tensor([length, max(1, length - 3)])
    padded = F.pad(waveform.unsqueeze(1), (0, -length % codec.frame_length))
    with torch.inference_mode():
        expected_codes = codec.quantizer(
            codec.encoder(padded), n_quantizers=n_quantizers
        ).codes
        codes, lengths = codec.encode(
            waveform, audio_lengths=audio_lengths, n_quantizers=n_quantizers
        )
        default_codes, default_lengths = codec.encode(
            waveform, n_quantizers=n_quantizers
        )

    assert torch.equal(codes, expected_codes)
    assert torch.equal(default_codes, expected_codes)
    assert torch.equal(lengths, torch.ceil(audio_lengths / codec.frame_length).long())
    assert default_lengths.tolist() == [math.ceil(length / codec.frame_length)]


def test_codec_encode_skips_reconstruction(
    codec: DAC, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = torch.randn(1, 17)
    with torch.inference_mode():
        expected_codes, expected_lengths = codec.encode(waveform, semantic_len=None)

    for module in (
        codec.quantizer.post_module,
        codec.quantizer.upsample,
        codec.decoder,
    ):
        monkeypatch.setattr(
            module, "forward", Mock(side_effect=AssertionError("reconstruction called"))
        )
    with torch.inference_mode():
        codes, lengths = codec.encode(waveform)

    assert torch.equal(codes, expected_codes)
    assert torch.equal(lengths, expected_lengths)


def test_codec_encode_preserves_training_and_kwargs(
    codec: DAC, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = torch.randn(2, 17)
    full_forward = Mock(wraps=codec.quantizer.forward)
    monkeypatch.setattr(codec.quantizer, "forward", full_forward)
    codec.train()
    with torch.no_grad():
        codec.encode(waveform, n_quantizers=1)
    assert full_forward.call_count == 1
    assert full_forward.call_args.args[1] == 1

    codec.eval()
    semantic_len = torch.tensor([5, 4])
    with torch.inference_mode():
        codec.encode(waveform, semantic_len=semantic_len)
    assert full_forward.call_count == 2
    assert full_forward.call_args.kwargs["semantic_len"] is semantic_len


def test_full_quantizer_still_reconstructs_and_backpropagates(
    quantizer: DownsampleResidualVectorQuantize,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantizer.train()
    post_forward = Mock(wraps=quantizer.post_module.forward)
    upsample_forward = Mock(wraps=quantizer.upsample.forward)
    monkeypatch.setattr(quantizer.post_module, "forward", post_forward)
    monkeypatch.setattr(quantizer.upsample, "forward", upsample_forward)
    latent = torch.randn(2, 8, 9, requires_grad=True)
    result = quantizer(latent)

    assert isinstance(result, VQResult)
    assert result.z.shape == latent.shape
    assert result.latents.shape[0] == latent.shape[0]
    assert result.latents.shape[-1] == result.codes.shape[-1]
    assert result.semantic_distill_z is None
    assert post_forward.call_count == upsample_forward.call_count == 1
    loss = result.z.square().mean() + result.commitment_loss + result.codebook_loss
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert quantizer.post_module[0].weight.grad is not None


def test_codes_only_preserves_checkpoint_state(codec: DAC) -> None:
    checkpoint = {
        name: parameter.detach().clone()
        for name, parameter in codec.state_dict().items()
    }
    waveform = torch.randn(2, 17)
    with torch.inference_mode():
        codes, lengths = codec.encode(waveform)

    assert codec.state_dict().keys() == checkpoint.keys()
    for name, parameter in codec.state_dict().items():
        assert torch.equal(parameter, checkpoint[name]), name
    codec.load_state_dict(checkpoint, strict=True)
    with torch.inference_mode():
        restored_codes, restored_lengths = codec.encode(waveform)
    assert torch.equal(restored_codes, codes)
    assert torch.equal(restored_lengths, lengths)
