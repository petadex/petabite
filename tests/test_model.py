import pytest

from petabite.model_module import BackboneFactory, ModelFactory
from petabite.model_module.backbone import BACKBONE_REGISTRY


def test_backbone_registered():
    assert "esmc" in BACKBONE_REGISTRY.keys()


def test_model_factory_returns_class():
    assert ModelFactory("esmc_activity") is not None
    assert BackboneFactory("esmc") is not None


@pytest.mark.xfail(reason="ESM-C backbone load is a stub", strict=True)
def test_backbone_forward_stub_raises():
    cls = BackboneFactory("esmc")
    backbone = cls(model_name="dummy", lora_r=8)
    backbone.forward(input_ids=None, attention_mask=None)
