from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from suave.modules.decoder import CountHead, Decoder, OrdinalHead, PosHead, RealHead
from suave.modules.distributions import (
    nll_gaussian,
    nll_lognormal,
    nll_ordinal,
    nll_poisson,
    ordinal_probabilities,
)
from suave.modules.encoder import EncoderMLP
from suave.sampling import sample_mixture_latents
from suave.types import Schema


def test_encoder_mlp_shapes():
    encoder = EncoderMLP(
        input_dim=5, latent_dim=3, hidden=(4,), dropout=0.0, n_components=2
    )
    x = torch.randn(7, 5)
    logits, mu, logvar = encoder(x)
    assert logits.shape == (7, 2)
    assert mu.shape == (7, 2, 3)
    assert logvar.shape == (7, 2, 3)
    assert torch.all(logvar <= EncoderMLP.LOGVAR_RANGE[1])
    assert torch.all(logvar >= EncoderMLP.LOGVAR_RANGE[0])


def test_sample_mixture_latents_outputs_assignments():
    logits = torch.tensor([0.5, 0.5])
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)
    latents, assignments = sample_mixture_latents(
        logits, mu, logvar, n_samples=4, device=torch.device("cpu")
    )
    assert latents.shape == (4, 3)
    assert assignments.shape == (4,)
    assert assignments.dtype == torch.long
    assert assignments.min() >= 0
    assert assignments.max() < 2


def test_real_head_output_shapes():
    head = RealHead(y_dim=2, n_components=3)
    batch = 4
    y = torch.randn(batch, 2)
    assignments = F.one_hot(torch.tensor([0, 1, 2, 1]), num_classes=3).float()
    x = torch.zeros(batch, 1)
    mask = torch.ones(batch, 1)
    stats = {"mean": 2.0, "std": 3.0}
    output = head(y=y, s=assignments, x=x, norm_stats=stats, mask=mask)
    assert output["log_px"].shape == (batch,)
    assert output["params"]["mean"].shape == (batch, 1)
    assert output["params"]["var"].shape == (batch, 1)
    sample = output["sample"]
    assert sample.shape == (batch, 1)


def test_decoder_forward_mixed_types():
    schema = Schema(
        {
            "age": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
            "income": {"type": "pos"},
            "visits": {"type": "count"},
            "severity": {"type": "ordinal", "n_classes": 3},
        }
    )
    decoder = Decoder(
        latent_dim=2, schema=schema, hidden=(4,), dropout=0.0, n_components=2
    )
    latents = torch.zeros(3, 2)
    assignments = F.one_hot(torch.tensor([0, 1, 0]), num_classes=2).float()
    data = {
        "real": {"age": torch.zeros(3, 1)},
        "pos": {"income": torch.zeros(3, 1)},
        "count": {"visits": torch.zeros(3, 1)},
        "cat": {
            "gender": nn.functional.one_hot(
                torch.tensor([0, 1, 0]), num_classes=2
            ).float()
        },
        "ordinal": {
            "severity": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
            )
        },
    }
    masks = {
        "real": {"age": torch.ones(3, 1)},
        "pos": {"income": torch.ones(3, 1)},
        "count": {"visits": torch.ones(3, 1)},
        "cat": {"gender": torch.ones(3, 1)},
        "ordinal": {"severity": torch.ones(3, 1)},
    }
    stats = {
        "age": {"mean": 0.0, "std": 1.0},
        "gender": {"categories": [0, 1]},
        "income": {"mean_log": 0.0, "std_log": 1.0},
        "visits": {"offset": 0.0},
        "severity": {"n_classes": 3},
    }
    output = decoder(latents, assignments, data, stats, masks)
    assert set(output["per_feature"]) == {
        "age",
        "gender",
        "income",
        "visits",
        "severity",
    }
    assert len(output["log_px"]) == 5
    assert output["per_feature"]["age"]["params"]["mean"].shape == (3, 1)
    assert output["per_feature"]["severity"]["sample"].shape == (3, 3)


def test_nll_gaussian_matches_closed_form():
    x = torch.zeros(4, 1)
    mu = torch.zeros(4, 1)
    var = torch.full((4, 1), 2.0)
    mask = torch.ones(4, 1)
    nll = nll_gaussian(x, mu, var, mask)
    expected = 0.5 * (math.log(2 * math.pi * 2.0))
    assert torch.allclose(nll, torch.full((4,), expected), atol=1e-5)


def test_pos_head_output_shapes():
    head = PosHead(y_dim=1, n_components=2)
    batch = 3
    y = torch.randn(batch, 1)
    assignments = F.one_hot(torch.tensor([0, 1, 0]), num_classes=2).float()
    x = torch.zeros(batch, 1)
    mask = torch.ones(batch, 1)
    stats = {"mean_log": 0.0, "std_log": 1.0}
    output = head(y=y, s=assignments, x=x, norm_stats=stats, mask=mask)
    assert output["log_px"].shape == (batch,)
    assert torch.isfinite(output["log_px"]).all()
    assert output["sample"].shape == (batch, 1)


def test_count_head_respects_offset():
    head = CountHead(y_dim=1, n_components=2)
    x = torch.log(torch.tensor([[2.0], [3.0]]))
    y = torch.randn(2, 1)
    assignments = F.one_hot(torch.tensor([0, 1]), num_classes=2).float()
    mask = torch.ones(2, 1)
    stats = {"offset": 0.0}
    output = head(y=y, s=assignments, x=x, norm_stats=stats, mask=mask)
    assert output["params"]["rate"].shape == (2, 1)
    assert torch.isfinite(output["log_px"]).all()


def test_ordinal_head_shapes():
    head = OrdinalHead(y_dim=1, n_components=2, n_classes=3)
    thermo = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y = torch.randn(2, 1)
    assignments = F.one_hot(torch.tensor([0, 1]), num_classes=2).float()
    mask = torch.ones(2, 1)
    output = head(y=y, s=assignments, x=thermo, norm_stats={}, mask=mask)
    assert output["params"]["probs"].shape == (2, 3)
    assert torch.isfinite(output["log_px"]).all()


def test_nll_lognormal_matches_closed_form():
    log_x = torch.log(torch.tensor([[2.0], [3.0]]))
    mean = torch.zeros_like(log_x)
    var = torch.ones_like(log_x)
    mask = torch.ones_like(log_x)
    nll = nll_lognormal(log_x, mean, var, mask)
    expected = 0.5 * (math.log(2 * math.pi) + log_x.squeeze(-1) ** 2) + log_x.squeeze(
        -1
    )
    assert torch.allclose(nll, expected, atol=1e-5)


def test_nll_poisson_matches_distribution():
    counts = torch.tensor([[0.0], [2.0]])
    rate = torch.full_like(counts, 1.5)
    mask = torch.ones_like(counts)
    nll = nll_poisson(counts, rate, mask)
    expected = -torch.distributions.Poisson(rate).log_prob(counts)
    assert torch.allclose(nll, expected.squeeze(-1))


def test_nll_ordinal_with_manual_probabilities():
    partition = torch.tensor([[0.2, 0.3]])
    mean = torch.tensor([[0.1]])
    probs, _ = ordinal_probabilities(partition, mean)
    mask = torch.ones(1)
    targets = torch.tensor([1])
    nll = nll_ordinal(probs, targets, mask)
    log_prob = torch.log(probs[:, 1])
    assert torch.allclose(nll, -log_prob)
