import torch

from petabite.model_module.heads import HEAD_REGISTRY


def test_regression_head_output_shape():
    cls = HEAD_REGISTRY.get("regression")
    head = cls(input_dim=16, output_dim=1)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 1)


def test_classification_head_output_shape():
    cls = HEAD_REGISTRY.get("classification")
    head = cls(input_dim=16, output_dim=3)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 3)
