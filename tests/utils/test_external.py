"""
Test external lazy import error handling.
"""

import builtins
import importlib
import sys

import pytest

from nerfstudio.utils import external


def test_lazy_error_raises_runtime_error_with_cause():
    """Verify _LazyError raises RuntimeError chained from the original exception."""
    original_exc = ModuleNotFoundError("No module named 'tinycudann'")
    lazy_mod = external._LazyError(("tinycudann", original_exc))

    lazy_attr = lazy_mod.Encoding
    with pytest.raises(RuntimeError, match="Could not load package tinycudann") as exc_info:
        lazy_attr(n_input_dims=3)
    assert exc_info.value.__cause__ is original_exc

    with pytest.raises(RuntimeError, match="Could not load package tinycudann") as attr_exc_info:
        _ = lazy_attr.nested
    assert attr_exc_info.value.__cause__ is original_exc


def test_tcnn_missing_raises_runtime_error(monkeypatch):
    """Verify importing nerfstudio.utils.external without tinycudann raises RuntimeError on attribute call."""
    monkeypatch.setitem(sys.modules, "tinycudann", None)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tinycudann":
            raise ModuleNotFoundError("No module named 'tinycudann'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reloaded = importlib.reload(external)

    try:
        assert not reloaded.TCNN_EXISTS
        assert isinstance(reloaded.tcnn_import_exception, ModuleNotFoundError)
        with pytest.raises(RuntimeError, match="Could not load package tinycudann") as exc_info:
            reloaded.tcnn.Encoding(n_input_dims=3)
        assert exc_info.value.__cause__ is reloaded.tcnn_import_exception
    finally:
        monkeypatch.undo()
        importlib.reload(external)
