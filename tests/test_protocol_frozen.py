"""The sim evaluation protocol is frozen at WO-02 (CLAUDE.md rule 4).

Every reported number cites `protocol_hash()`. Changing `src/eval/protocol.py` changes the hash and
fails this test on purpose: add a new protocol version (a new file and a new pinned hash) instead.
"""

from src.eval.protocol import PROTOCOL_VERSION, protocol_hash

FROZEN = {
    # frozen 2026-10-26 at WO-02; used by the WO-02 SFT sweeps and evals
    "0.1-draft": "e9ce5bb47a88da03d3cad77c4084b32d7184eeb2d30efa4f93f6f487be7e517c",
}


def test_protocol_is_frozen() -> None:
    assert PROTOCOL_VERSION in FROZEN, f"protocol {PROTOCOL_VERSION!r} has no pinned hash"
    assert protocol_hash() == FROZEN[PROTOCOL_VERSION]
