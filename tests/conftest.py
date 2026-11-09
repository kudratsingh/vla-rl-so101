"""Shared pytest setup: tests marked `needs_render` are skipped where SAPIEN has no Vulkan device.

GitHub's ubuntu runners have no GPU and Mesa's software Vulkan is not a device SAPIEN accepts, so
image-producing envs cannot be built there; the Mac (MoltenVK) and GPU boxes run these tests.

The robot tests' ``robot_fakes.real_cfg()`` sees configs/real.yaml with the WO-06 bring-up values
(port, joint limits, intrinsics, units, leader) reset to the shipped placeholders
(tests/real_fixture.py), so calibrating the real arm does not break the suite.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Iterator

import pytest
from real_fixture import load_config_placeholders


@pytest.fixture(autouse=True)
def _real_yaml_placeholders(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    fakes = sys.modules.get("robot_fakes")
    if fakes is not None:  # robot_fakes.real_cfg() calls its module-level load_config
        monkeypatch.setattr(fakes, "load_config", load_config_placeholders)
    prompts = sys.modules.get("src.robot.prompts")
    if prompts is not None:  # process-wide signal flags: never leak between tests
        prompts.INTERRUPTS.reset()
    yield
    if prompts is not None:
        prompts.INTERRUPTS.reset()


@functools.cache
def _can_render() -> bool:
    from src.env import _mac_vulkan_defaults

    _mac_vulkan_defaults()
    try:
        import sapien
        import torch

        sapien.render.RenderSystem(sapien.Device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:  # noqa: BLE001 - any failure means no usable render device
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any("needs_render" in item.keywords for item in items) or _can_render():
        return
    skip = pytest.mark.skip(reason="no Vulkan render device for SAPIEN (e.g. GPU-less CI runner)")
    for item in items:
        if "needs_render" in item.keywords:
            item.add_marker(skip)
