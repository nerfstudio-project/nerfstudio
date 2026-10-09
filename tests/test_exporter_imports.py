"""Tests for exporter dependency imports."""

import importlib
import sys


def test_exporter_modules_do_not_eagerly_import_pymeshlab(monkeypatch):
    """Exporter modules should import without initializing pymeshlab."""
    monkeypatch.setitem(sys.modules, "pymeshlab", None)

    modules = [
        "nerfstudio.exporter.exporter_utils",
        "nerfstudio.exporter.tsdf_utils",
    ]
    for module in modules:
        sys.modules.pop(module, None)

    for module in modules:
        importlib.import_module(module)

    assert sys.modules["pymeshlab"] is None
