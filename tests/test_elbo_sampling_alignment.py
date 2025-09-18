import torch
from torch.nn import functional as F

from suave.model import SUAVE
from suave.modules.decoder import Decoder
from suave.modules import losses
from suave.types import Schema


def _make_decoder() -> Decoder:
    schema = Schema({"feature": {"type": "real", "y_dim": 1}})
    decoder = Decoder(
        latent_dim=2, schema=schema, hidden=(), dropout=0.0, n_components=2
    )
    with torch.no_grad():
        decoder.y_projection.weight.copy_(torch.tensor([[0.8, -0.2]]))
        decoder.y_projection.bias.zero_()
        head = decoder.heads["feature"]
        head.mean_layer.weight.copy_(torch.tensor([[0.5, -0.3, 0.4]]))
        head.var_layer.weight.copy_(torch.tensor([[0.2, -0.1]]))
    return decoder


def test_sampled_elbo_matches_expectation_zero_temperature_limit() -> None:
    torch.manual_seed(0)
    decoder = _make_decoder()

    batch = 2
    x = torch.tensor([[0.5], [-1.0]], dtype=torch.float32)
    mask = torch.ones(batch, 1)
    data = {"real": {"feature": x}, "pos": {}, "count": {}, "cat": {}, "ordinal": {}}
    masks = {
        "real": {"feature": mask},
        "pos": {},
        "count": {},
        "cat": {},
        "ordinal": {},
    }
    norm_stats = {"feature": {"mean": 0.0, "std": 1.0}}

    component_logits = torch.tensor([[40.0, -40.0], [-40.0, 40.0]])
    component_mu = torch.tensor(
        [
            [[0.1, -0.2], [0.3, -0.1]],
            [[-0.4, 0.2], [0.6, -0.3]],
        ],
        dtype=torch.float32,
    )
    component_logvar = torch.full((batch, 2, 2), -1e6, dtype=torch.float32)
    posterior_probs = torch.softmax(component_logits, dim=-1)

    prior_logits = torch.zeros(2, dtype=torch.float32)
    prior_mu = torch.zeros(2, 2, dtype=torch.float32)
    prior_logvar = torch.zeros(2, 2, dtype=torch.float32)

    component_indices = component_logits.argmax(dim=-1)
    assignments = F.one_hot(component_indices, num_classes=2).float()
    selected_mu = SUAVE._gather_component_parameters(component_mu, component_indices)
    selected_logvar = SUAVE._gather_component_parameters(
        component_logvar, component_indices
    )
    z_sample = SUAVE._reparameterize(selected_mu, selected_logvar)

    decoder_out_new = decoder(z_sample, assignments, data, norm_stats, masks)
    recon_new = losses.sum_reconstruction_terms(decoder_out_new["log_px"])

    prior_logits_batch = prior_logits.expand_as(component_logits)
    categorical_kl = losses.kl_categorical(component_logits, prior_logits_batch)
    prior_mu_selected = prior_mu.index_select(0, component_indices)
    prior_logvar_selected = prior_logvar.index_select(0, component_indices)
    gaussian_kl_new = losses.kl_normal_vs_normal(
        selected_mu, selected_logvar, prior_mu_selected, prior_logvar_selected
    )
    elbo_new = recon_new - (categorical_kl + gaussian_kl_new)

    log_px_components = []
    for idx in range(assignments.size(-1)):
        component_assignments = F.one_hot(
            torch.full((batch,), idx, dtype=torch.long),
            num_classes=assignments.size(-1),
        ).float()
        z_component = SUAVE._reparameterize(
            component_mu[:, idx, :], component_logvar[:, idx, :]
        )
        decoder_out = decoder(
            z_component, component_assignments, data, norm_stats, masks
        )
        log_px_components.append(losses.sum_reconstruction_terms(decoder_out["log_px"]))
    log_px_stack = torch.stack(log_px_components, dim=0)
    recon_expectation = (posterior_probs.t() * log_px_stack).sum(dim=0)
    gaussian_kl_expectation = losses.kl_normal_mixture(
        component_mu, component_logvar, prior_mu, prior_logvar, posterior_probs
    )
    elbo_expectation = recon_expectation - (categorical_kl + gaussian_kl_expectation)

    torch.testing.assert_close(elbo_new, elbo_expectation, atol=1e-5, rtol=1e-4)
