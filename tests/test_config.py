from anima_slam_gs3lam.config import default_config, load_config


def test_default_config_uses_correct_module_identity():
    config = default_config()
    assert config.project.name == "anima-slam-gs3lam"
    assert config.project.paper_arxiv == "2603.27781"
    assert config.project.python_version == "3.11"


def test_runtime_config_prefers_paper_defaults_for_scannet():
    config = default_config()
    runtime = config.build_runtime_config("scannet")
    assert runtime.tracking.iterations == 100
    assert runtime.mapping.iterations == 30


def test_runtime_config_can_apply_repo_overrides_for_scannet():
    config = default_config()
    runtime = config.build_runtime_config("scannet", use_repo_overrides=True)
    assert runtime.tracking.iterations == 200
    assert runtime.mapping.iterations == 60


def test_load_config_reads_default_toml():
    config = load_config()
    assert config.dataset_presets["replica"].sequence == "office0"
    assert config.dataset_presets["tum"].camera.crop_edge == 8
