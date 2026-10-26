from src.config import config_hash, load_config


def test_presets_load_and_inherit() -> None:
    tiny = load_config("tiny")
    assert tiny["device"] == "cpu"
    rl = load_config("rl")
    assert rl["env"]["image_size"] == 128  # inherited from sim.yaml
    assert rl["rl"]["algorithm"] == "grpo"
    assert config_hash(rl) == config_hash(load_config("rl"))
