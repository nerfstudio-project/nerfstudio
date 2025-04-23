"""
Tests for the nerfstudio.plugins.registry module.
"""

import os
import sys
from dataclasses import dataclass, field

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins import registry, registry_dataparser
from nerfstudio.plugins.registry_dataparser import DataParserConfig, DataParserSpecification, discover_dataparsers
from nerfstudio.plugins.types import MethodSpecification

if sys.version_info < (3, 10):
    import importlib_metadata
else:
    from importlib import metadata as importlib_metadata


@dataclass
class TestConfigClass(MethodSpecification):
    config: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            method_name="test-method",
            pipeline=VanillaPipelineConfig(),
            optimizers={},
        )
    )
    description: str = "Test description"


def TestConfigFunc():
    return TestConfigClass()


TestConfig = TestConfigClass()


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


def _test_discover_methods_from_environment_variable(method):
    """Tests if a custom method from env variable gets properly registered using the discover_methods method"""
    old_env = None
    try:
        old_env = os.environ.get("NERFSTUDIO_METHOD_CONFIGS", None)
        method_key = f"{method}-env"
        os.environ["NERFSTUDIO_METHOD_CONFIGS"] = f"{method_key}=test_registry:{method}"

        # Discover plugins
        methods, _ = registry.discover_methods()
        assert method_key in methods
        config = methods[method_key]
        assert isinstance(config, TrainerConfig)
    finally:
        # Revert mock
        if old_env is not None:
            os.environ["NERFSTUDIO_METHOD_CONFIGS"] = old_env
        else:
            del os.environ["NERFSTUDIO_METHOD_CONFIGS"]


def test_discover_methods_from_environment_variable_class():
    """Tests if a custom method class from env variable gets properly registered using the discover_methods method"""
    _test_discover_methods_from_environment_variable("TestConfigClass")


def test_discover_methods_from_environment_variable_function():
    """Tests if a custom method function from env variable gets properly registered using the discover_methods method"""
    _test_discover_methods_from_environment_variable("TestConfigFunc")


def test_discover_methods_from_environment_variable_instance():
    """Tests if a custom method instance from env variable gets properly registered using the discover_methods method"""
    _test_discover_methods_from_environment_variable("TestConfig")


@dataclass
class TestDataparserConfigClass(DataParserSpecification):
    config: DataParserConfig = field(default_factory=DataParserConfig)
    description: str = "Test description"


def TestDataparserConfigFunc():
    return TestDataparserConfigClass()


TestDataparserConfig = TestDataparserConfigClass()


def test_discover_dataparser():
    """Tests if a custom method gets properly registered using the discover_dataparsers method"""
    entry_points_backup = registry_dataparser.entry_points

    def entry_points(group=None):
        return importlib_metadata.EntryPoints(
            [
                importlib_metadata.EntryPoint(
                    name="test", value="test_registry:TestDataparserConfig", group="nerfstudio.dataparser_configs"
                )
            ]
        ).select(group=group)

    try:
        # Mock importlib entry_points
        registry_dataparser.entry_points = entry_points

        # Discover plugins
        dataparsers, _ = discover_dataparsers()
        assert "test" in dataparsers
        config = dataparsers["test"]
        assert isinstance(config, DataParserConfig)
    finally:
        # Revert mock
        registry_dataparser.entry_points = entry_points_backup


def _test_discover_dataparsers_from_environment_variable(method):
    """Tests if a custom method from env variable gets properly registered using the discover_dataparsers method"""
    old_env = None
    try:
        old_env = os.environ.get("NERFSTUDIO_DATAPARSER_CONFIGS", None)
        method_key = f"{method}-env"
        os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = f"{method_key}=test_registry:{method}"

        # Discover plugins
        dataparsers, _ = discover_dataparsers()
        assert method_key in dataparsers
        config = dataparsers[method_key]
        assert isinstance(config, DataParserConfig)
    finally:
        # Revert mock
        if old_env is not None:
            os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = old_env
        else:
            del os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"]


def test_discover_dataparsers_from_environment_variable_class():
    """Tests if a custom dataparser class from env variable gets properly registered using the discover_methods method"""
    _test_discover_dataparsers_from_environment_variable("TestDataparserConfigClass")


def test_discover_dataparsers_from_environment_variable_function():
    """Tests if a custom dataparser function from env variable gets properly registered using the discover_methods method"""
    _test_discover_dataparsers_from_environment_variable("TestDataparserConfigFunc")


def test_discover_dataparsers_from_environment_variable_instance():
    """Tests if a custom dataparser instance from env variable gets properly registered using the discover_methods method"""
    _test_discover_dataparsers_from_environment_variable("TestDataparserConfig")
