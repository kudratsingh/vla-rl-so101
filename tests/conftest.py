"""Shared pytest setup: tests marked `needs_render` are skipped where SAPIEN has no Vulkan device.

GitHub's ubuntu runners have no GPU and Mesa's software Vulkan is not a device SAPIEN accepts, so
image-producing envs cannot be built there; the Mac (MoltenVK) and GPU boxes run these tests.
"""

from __future__ import annotations

import functools

import pytest


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
