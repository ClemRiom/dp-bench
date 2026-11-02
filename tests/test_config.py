import pytest
import yaml

from conftest import tiny_config
from dpbench.core.config import (
    ConfigError,
    PrivacyConfig,
    config_from_dict,
    load_config,
    load_config_text,
    validate_plugins,
)
from dpbench.experiments.runner import plan_jobs


def test_unknown_key_gets_a_suggestion(tmp_path):
    data = tiny_config()
    data["trainin"] = data.pop("training")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="Did you mean 'training'"):
        load_config(path)


def test_unknown_mechanism_and_parameter():
    config = config_from_dict(tiny_config(mechanisms=[{"name": "dp_sgd"}]))
    with pytest.raises(ConfigError, match="Did you mean"):
        validate_plugins(config)
    config = config_from_dict(tiny_config(mechanisms=[{"name": "dpsgd", "params": {"learning_rate": 0.1}}]))
    with pytest.raises(ConfigError, match="no parameter 'learning_rate'"):
        validate_plugins(config)


def test_duplicate_labels_are_rejected():
    with pytest.raises(ConfigError, match="label"):
        config_from_dict(tiny_config(mechanisms=["dpsgd", "dpsgd"]))
    config = config_from_dict(
        tiny_config(
            mechanisms=[
                {"name": "dpsgd", "label": "dpsgd_c1", "params": {"max_grad_norm": 1.0}},
                {"name": "dpsgd", "label": "dpsgd_c01", "params": {"max_grad_norm": 0.1}},
            ]
        )
    )
    assert [m.label for m in config.mechanisms] == ["dpsgd_c1", "dpsgd_c01"]


@pytest.mark.parametrize("epsilons", [[0], [-1.0], [], "many"])
def test_invalid_budgets(epsilons):
    with pytest.raises(ConfigError):
        config_from_dict(tiny_config(privacy={"epsilons": epsilons}))


def test_delta_auto_is_one_over_n():
    assert PrivacyConfig(delta="auto").resolve_delta(1000) == pytest.approx(1e-3)
    assert PrivacyConfig(delta=1e-5).resolve_delta(1000) == 1e-5
    with pytest.raises(ConfigError):
        config_from_dict(tiny_config(privacy={"epsilons": [1], "delta": 2.0}))


def test_mechanism_training_overrides():
    config = config_from_dict(tiny_config(mechanisms=[{"name": "dpsgd", "training": {"batch_size": 7}}]))
    assert config.training_for(config.mechanisms[0]).batch_size == 7
    assert config.training.batch_size == 32
    with pytest.raises(ConfigError):
        config_from_dict(tiny_config(mechanisms=[{"name": "dpsgd", "training": {"batchsize": 7}}]))


def test_round_trip_through_yaml():
    config = config_from_dict(tiny_config())
    again = load_config_text(config.to_yaml())
    assert again.to_dict() == config.to_dict()


def test_plan_trains_the_baseline_once_per_seed():
    config = config_from_dict(tiny_config(seeds=[0, 1], privacy={"epsilons": [1, 2, 4]}))
    jobs = plan_jobs(config)
    assert len(jobs) == 2 + 3 * 2
    assert [j.run_id for j in jobs][:2] == ["non_private|eps=inf|seed=0", "non_private|eps=inf|seed=1"]
