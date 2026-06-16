from petabite.utils import Registry, set_seed


def test_registry_register_and_get():
    reg: Registry = Registry("demo")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert "a" in reg.keys()


def test_registry_unknown_key_raises():
    reg: Registry = Registry("demo")
    try:
        reg.get("missing")
    except KeyError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_set_seed_is_deterministic():
    import random
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b
