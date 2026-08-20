import pytest
from config import TaskFConfig, ModelConfig


def test_defaults_are_valid():
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"))
    assert cfg.outer_steps == 200
    assert cfg.inner_steps == 5
    assert cfg.bound_embeddings is True
    assert cfg.posttrain is False


def test_rejects_random_forest():
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="random_forest"))


def test_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), outer_steps=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), inner_steps=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), num_restarts=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), validation_split=1.5)
