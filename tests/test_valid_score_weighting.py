import pytest
import torch

from flow_matching.score_gradient_rl import valid_score_weights


def test_equal_scores_preserve_mean_and_loss_gradient():
    scores = torch.full((8,), 60.0, requires_grad=True)
    losses = torch.arange(8.0, requires_grad=True)
    weights = valid_score_weights(scores, reference_max=scores.max(), tau=7.5, epsilon=0.01)
    objective = (weights * losses).sum() / weights.sum()
    objective.backward()
    torch.testing.assert_close(objective, losses.mean())
    torch.testing.assert_close(losses.grad, torch.full((8,), 1 / 8))
    assert scores.grad is None


def test_weights_match_approved_formula_and_stay_finite():
    scores = torch.tensor([-10000.0, 50.0, 60.0, 70.0])
    weights = valid_score_weights(scores, reference_max=scores.max(), tau=7.5, epsilon=0.01)
    torch.testing.assert_close(weights, torch.exp((scores - 70) / 7.5) + 0.01)
    assert torch.isfinite(weights).all()
    assert torch.all(weights[1:] > weights[:-1])


@pytest.mark.parametrize("tau,epsilon", [(0, 0.01), (7.5, 0), (float("nan"), 0.01)])
def test_invalid_hyperparameters_rejected(tau, epsilon):
    with pytest.raises(ValueError):
        valid_score_weights(torch.ones(2), reference_max=torch.tensor(1.0), tau=tau, epsilon=epsilon)
