from __future__ import annotations

import math

import torch
from torch import nn

from suave.modules.decoder import Decoder, RealHead
from suave.modules.distributions import nll_gaussian
from suave.modules.encoder import EncoderMLP
from suave.types import Schema


def test_encoder_mlp_shapes():
    encoder = EncoderMLP(input_dim=5, latent_dim=3, hidden=(4,), dropout=0.0)
    x = torch.randn(7, 5)
    mu, logvar = encoder(x)
    assert mu.shape == (7, 3)
    assert logvar.shape == (7, 3)
    assert torch.all(logvar <= EncoderMLP.LOGVAR_RANGE[1])
    assert torch.all(logvar >= EncoderMLP.LOGVAR_RANGE[0])


def test_real_head_output_shapes():
    head = RealHead()
    x = torch.zeros(2, 1)
    params = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    mask = torch.ones(2, 1)
    stats = {"mean": 2.0, "std": 3.0}
    output = head(x=x, params=params, norm_stats=stats, mask=mask)
    assert output["log_px"].shape == (2,)
    assert output["params"]["mean"].shape == (2, 1)
    assert output["params"]["var"].shape == (2, 1)
    sample = output["sample"]
    assert sample.shape == (2, 1)


def test_decoder_forward_real_cat():
    schema = Schema(
        {
            "age": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
        }
    )
    decoder = Decoder(latent_dim=2, schema=schema, hidden=(4,), dropout=0.0)
    latents = torch.zeros(3, 2)
    data = {
        "real": {"age": torch.zeros(3, 1)},
        "cat": {
            "gender": nn.functional.one_hot(
                torch.tensor([0, 1, 0]), num_classes=2
            ).float()
        },
    }
    masks = {
        "real": {"age": torch.ones(3, 1)},
        "cat": {"gender": torch.ones(3, 1)},
    }
    stats = {
        "age": {"mean": 0.0, "std": 1.0},
        "gender": {"categories": [0, 1]},
    }
    output = decoder(latents, data, stats, masks)
    assert set(output["per_feature"]) == {"age", "gender"}
    assert len(output["log_px"]) == 2
    assert output["per_feature"]["age"]["params"]["mean"].shape == (3, 1)


def test_nll_gaussian_matches_closed_form():
    x = torch.zeros(4, 1)
    mu = torch.zeros(4, 1)
    var = torch.full((4, 1), 2.0)
    mask = torch.ones(4, 1)
    nll = nll_gaussian(x, mu, var, mask)
    expected = 0.5 * (math.log(2 * math.pi * 2.0))
    assert torch.allclose(nll, torch.full((4,), expected), atol=1e-5)
