"""Set one value in a YAML config file in place, keeping every comment and all other lines.

Used by the WO-06 scripts to write calibration results (port, joint limits, intrinsics) into
``configs/real.yaml``. Line-based on purpose (no ruamel dependency): the key must be a plain
``key:`` mapping entry; its old value (inline, or an indented block under it) is replaced by a
one-line flow-style YAML value; its trailing ``# comment`` and comment lines are kept. The result
is re-parsed and checked before the file is written.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_\-]+):(?P<rest>.*)$")


def _flow(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, width=10**9, sort_keys=False).strip()
    text = text.removesuffix("\n...").removesuffix("...").strip()
    if "\n" in text:
        raise ValueError(f"value does not fit on one line: {value!r}")
    return text


def _split_comment(rest: str) -> tuple[str, str]:
    """(value, ' # comment' incl. leading spaces) for the text after ``key:``."""
    m = re.search(r"\s+#", rest)
    if m is None:
        return rest, ""
    return rest[: m.start()], rest[m.start() :]


def set_yaml_value(text: str, keys: Sequence[str], value: Any) -> str:
    """Return ``text`` with the mapping entry at ``keys`` set to ``value``."""
    lines = text.splitlines(keepends=False)
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY.match(line)
        if m is None:
            continue
        indent = len(m["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m["key"]))
        if [k for _, k in stack] != list(keys):
            continue
        old_inline, comment = _split_comment(m["rest"])
        lines[i] = f"{m['indent']}{m['key']}: {_flow(value)}{comment}"
        if not old_inline.strip():
            # the old value was an indented block under the key: drop its non-comment lines
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                j += 1
            lines[i + 1 : j] = [
                ln for ln in lines[i + 1 : j] if not ln.strip() or ln.lstrip().startswith("#")
            ]
        out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        node: Any = yaml.safe_load(out)
        for k in keys:
            node = node[k]
        if node != yaml.safe_load(_flow(value)):
            raise RuntimeError(f"YAML edit of {'.'.join(keys)} did not round-trip")
        return out
    raise KeyError(f"key {'.'.join(keys)} not found")


def update_yaml_file(path: str | Path, keys: Sequence[str], value: Any) -> None:
    p = Path(path)
    p.write_text(set_yaml_value(p.read_text(), keys, value))
