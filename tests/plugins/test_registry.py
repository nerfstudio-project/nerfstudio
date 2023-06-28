"""
Tests for the nerfstudio.plugins.registry module.
"""
import os
import sys

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins import registry
from nerfstudio.plugins.types import MethodSpecification

if sys.version_info < (3, 10):
    import importlib_metadata
else:
    from importlib import metadata as importlib_metadata


TestConfig = MethodSpecification(
    config=TrainerConfig(
        method_name="test-method",
        pipeline=VanillaPipelineConfig(),
        optimizers={},
    ),
    description="Test description",
)


def test_discover_methods():
    """Tests if a custom method gets properly registered using the discover_methods method"""
    entry_points_backup = registry.entry_points

    def entry_points(group=None):
        return importlib_metadata.EntryPoints(
            [
                importlib_metadata.EntryPoint(
                    name="test", value="test_registry:TestConfig", group="nerfstudio.method_configs"
                )
            ]
        ).select(group=group)

    try:
        # Mock importlib entry_points
        registry.entry_points = entry_points

        # Discover plugins
        methods, _ = registry.discover_methods()
        assert "test-method" in methods
        config = methods["test-method"]
        assert isinstance(config, TrainerConfig)
    finally:
        # Revert mock
        registry.entry_points = entry_points_backup


def test_discover_methods_from_environment_variable():
    """Tests if a custom method from env variable gets properly registered using the discover_methods method"""
    old_env = None
    try:
        old_env = os.environ.get("NERFSTUDIO_METHOD_CONFIGS", None)
        os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "test-method-env=test_registry:TestConfig"

        # Discover plugins
        methods, _ = registry.discover_methods()
        assert "test-method-env" in methods
        config = methods["test-method-env"]
        assert isinstance(config, TrainerConfig)
    finally:
        # Revert mock
        if old_env is not None:
            os.environ["NERFSTUDIO_METHOD_CONFIGS"] = old_env
        else:
            del os.environ["NERFSTUDIO_METHOD_CONFIGS"]
