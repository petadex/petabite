import importlib.util
from pathlib import Path


def _load(name: str):
    path = Path(__file__).parent.parent / "src" / "petabite" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_modules_expose_main():
    for name in ["train", "predict", "mlm_finetune"]:
        mod = _load(name)
        assert hasattr(mod, "main"), f"{name}.py must define main()"
