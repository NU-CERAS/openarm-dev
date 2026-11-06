from types import SimpleNamespace

import pytest

from openarm_sim import compute_guard as guard


def test_small_machine_can_override_default_reserve(monkeypatch):
    monkeypatch.setattr(guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * guard.GIB))
    monkeypatch.delitem(guard.sys.modules, "torch", raising=False)
    monkeypatch.delenv("OPENARM_RAM_RESERVE_GIB", raising=False)
    with pytest.raises(MemoryError, match="8 GiB"):
        guard.check_resources()
    monkeypatch.setenv("OPENARM_RAM_RESERVE_GIB", "1")
    guard.check_resources()
    monkeypatch.setattr(guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=guard.GIB // 2))
    with pytest.raises(MemoryError, match="1 GiB"):
        guard.check_resources()
    monkeypatch.setenv("OPENARM_RAM_RESERVE_GIB", "0")
    guard.check_resources()


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_reserve_rejected(monkeypatch, value):
    monkeypatch.setenv("OPENARM_RAM_RESERVE_GIB", value)
    with pytest.raises(ValueError):
        guard.ram_reserve_bytes()


def test_ram_override_retains_gpu_reserve(monkeypatch):
    monkeypatch.setenv("OPENARM_RAM_RESERVE_GIB", "0")
    monkeypatch.setattr(guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=guard.GIB))
    cuda = SimpleNamespace(is_initialized=lambda: True, mem_get_info=lambda: (guard.GIB, 16 * guard.GIB))
    monkeypatch.setitem(guard.sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(MemoryError, match="GPU"):
        guard.check_resources()
