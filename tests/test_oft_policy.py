"""OpenVLA-OFT wrapper on a tiny random model (CPU, no weights downloaded).

Checks: token mapping == upstream OFT, our single forward == SimpleVLA-RL's rollout
(`generate_action_verl`) and log-prob (`forward`) paths, bidirectional attention, prompt/token
layout, padding isolation, sample/log_prob/SFT consistency, grad checkpointing, save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.config import load_config
from src.policy.base import SFTBatch, VLAPolicy
from src.policy.oft import model as oft_model
from src.policy.oft.actions import ACTION_VOCAB_BEGIN, ActionBins, NormBounds
from src.policy.oft.preprocess import PAD_ID, SPACE_ID, STOP_ID, OFTPreprocessor, ToyPromptTokenizer
from src.policy.openvla_oft import OpenVLAOFTPolicy, build_policy

INSTR = [
    "pick up the red cube and place it in the bowl",
    "push the red cube to the target",
    "stack the red cube on the blue cube",
]
BOUNDS = NormBounds.from_lists([-1.3, -1.7, -1.1, -1.6, -1.6, -0.2], [1.3, 0.9, 1.7, 1.7, 1.6, 0.7])


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return load_config("tiny", "configs/sft_tiny.yaml")


@pytest.fixture(scope="module")
def policy(cfg: dict[str, Any]) -> OpenVLAOFTPolicy:
    return build_policy(cfg, "cpu")


def _obs(b: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.randint(0, 256, (b, 96, 96, 3), dtype=torch.uint8, generator=g),
        "state": torch.rand(b, 6, generator=g) - 0.5,
    }


def _pre() -> OFTPreprocessor:
    return OFTPreprocessor(
        ToyPromptTokenizer(),
        action_bounds=BOUNDS,
        proprio_bounds=BOUNDS,
        chunk_len=8,
        action_dim=6,
        image_size=64,
    )


def test_action_bins_match_upstream_oft() -> None:
    """Upstream: id = 32000 - digitize(a, linspace(-1,1,256)); value = centers[clip(32000-id-1)]."""
    bins = ActionBins()
    a = np.concatenate([np.linspace(-1.2, 1.2, 4001), [-1.0, 1.0, 0.0]])
    edges = np.linspace(-1, 1, 256)
    centers = (edges[:-1] + edges[1:]) / 2
    up_ids = 32000 - np.digitize(np.clip(a, -1, 1), edges)
    up_val = centers[np.clip(32000 - up_ids - 1, 0, 254)]
    tok = bins.encode(torch.tensor(a))
    assert torch.equal(tok + ACTION_VOCAB_BEGIN, torch.tensor(up_ids))
    assert np.allclose(bins.decode(tok).numpy(), up_val, atol=1e-7)
    assert int(tok.min()) >= 0 and int(tok.max()) <= 255


def test_token_layout_and_padding() -> None:
    pre = _pre()
    obs = _obs(2)
    acts = torch.zeros(2, 8, 6)
    inp = pre.build_inputs(INSTR[:2], obs["image"], obs["state"], acts)
    for i, instr in enumerate(INSTR[:2]):
        p = pre.prompt_ids(instr)
        assert p[0] == 1 and p[-1] == SPACE_ID
        s = int(inp.action_start[i])
        assert s == len(p)
        row = inp.input_ids[i].tolist()
        assert row[:s] == p
        assert inp.action_mask[i, s : s + 48].all() and int(inp.action_mask[i].sum()) == 48
        assert row[s + 48] == STOP_ID
        assert all(t == PAD_ID for t in row[s + 49 :])
        assert int(inp.attention_mask[i].sum()) == s + 49
    assert inp.pixel_values.shape == (2, 6, 64, 64)
    assert inp.targets is not None and inp.targets.shape == (2, 48)
    # instructions are lower-cased, as upstream does
    assert pre.prompt_ids("PUSH the red cube to the target") == pre.prompt_ids(INSTR[1])


def test_real_tokenizer_layout() -> None:
    """With the actual Llama-2 tokenizer (2 MB download; skipped offline): the prompt ends
    '\\nOut:' + '▁' exactly as OFT's training tokenisation ("Out: " + action chars)."""
    try:
        from src.policy.oft.preprocess import HFPromptTokenizer

        tok = HFPromptTokenizer("openvla/openvla-7b", "47a0ec7fc4ec123775a391911046cf33cf9ed83f")
    except Exception as exc:  # noqa: BLE001  (network / hub errors of any kind -> skip)
        pytest.skip(f"tokenizer unavailable: {exc}")
    pre = OFTPreprocessor(
        tok, action_bounds=BOUNDS, proprio_bounds=BOUNDS, chunk_len=8, action_dim=6, image_size=64
    )
    ids = pre.prompt_ids("push the cube to the target")
    assert ids[:3] == [1, 512, 29901] and ids[-4:] == [13, 3744, 29901, SPACE_ID]
    train = tok.tok("In: x?\nOut: " + tok.tok.decode([31800, 31999]) + "</s>").input_ids
    assert train[-5:] == [29901, SPACE_ID, 31800, 31999, STOP_ID]


def test_parity_with_simplevla_rl_paths(cfg: dict[str, Any]) -> None:
    """Our action_logits == upstream `forward` (log-prob path) and `generate_action_verl`."""
    model = oft_model.load_model(
        checkpoint=None,
        revision=None,
        action_bounds=BOUNDS,
        proprio_bounds=BOUNDS,
        dtype=torch.float32,
        tiny_seed=0,
    ).eval()
    pre = _pre()
    obs = _obs(3, seed=1)
    with torch.no_grad():
        inp = pre.build_inputs(INSTR, obs["image"], obs["state"])
        ours = oft_model.action_logits(model, inp, 48)
        # upstream log-prob path: equal-length prompts only (it asserts that upstream) -> per sample
        for i in range(3):
            n = int(inp.action_start[i])
            up = model(
                input_ids=inp.input_ids[i : i + 1, :n],
                attention_mask=inp.attention_mask[i : i + 1, :n],
                pixel_values=inp.pixel_values[i : i + 1],
                proprio=inp.proprio[i : i + 1],
            )[..., -320:-64]
            assert torch.allclose(up, ours[i : i + 1], atol=1e-5), i
        # upstream rollout path: a right-padded batch of mixed-length prompts, greedy
        n_max = int(inp.action_start.max())
        ids = torch.full((3, n_max), PAD_ID, dtype=torch.long)
        am = torch.zeros((3, n_max), dtype=torch.long)
        for i in range(3):
            n = int(inp.action_start[i])
            ids[i, :n], am[i, :n] = inp.input_ids[i, :n], 1
        actions, vocab_ids = model.generate_action_verl(
            input_ids=ids,
            attention_mask=am,
            pixel_values=inp.pixel_values,
            proprio=inp.proprio,
            padding_idx=PAD_ID,
            do_sample=False,
            temperature=1.0,
            unnorm_key="so101",
        )
    tokens = ours.argmax(-1)
    assert torch.equal(vocab_ids - ACTION_VOCAB_BEGIN, tokens)
    assert np.allclose(actions, pre.decode_actions(tokens).numpy(), atol=1e-5)


def test_attention_is_bidirectional_and_hook_required() -> None:
    pre = _pre()
    obs = _obs(1, seed=2)
    inp = pre.build_inputs(INSTR[:1], obs["image"], obs["state"])
    stop_pos = int(inp.action_start[0]) + 48
    changed = inp.input_ids.clone()
    changed[0, stop_pos] = 5  # a token AFTER every action slot
    from dataclasses import replace

    probe = replace(inp, input_ids=changed)
    model = oft_model.load_model(
        checkpoint=None,
        revision=None,
        action_bounds=BOUNDS,
        proprio_bounds=BOUNDS,
        dtype=torch.float32,
        tiny_seed=0,
    ).eval()
    torch.manual_seed(0)
    causal = oft_model.OpenVLAForActionPrediction(oft_model.tiny_config(BOUNDS, BOUNDS)).eval()
    causal.vision_backbone.set_num_images_in_input(1)
    with torch.no_grad():
        d_bi = oft_model.action_logits(model, inp, 48) - oft_model.action_logits(model, probe, 48)
        d_ca = oft_model.action_logits(causal, inp, 48) - oft_model.action_logits(causal, probe, 48)
    assert float(d_bi.abs().max()) > 1e-6  # later tokens influence the action slots
    assert float(d_ca.abs().max()) == 0.0  # without the hook the stock Llama is causal


def test_sdpa_is_enforced() -> None:
    cfg = oft_model.tiny_config(BOUNDS, BOUNDS)
    cfg._attn_implementation = "eager"
    model = oft_model.OpenVLAForActionPrediction(cfg)
    with pytest.raises(RuntimeError, match="sdpa"):
        oft_model.attach_bidirectional_attention(model)


def test_padding_does_not_leak(policy: OpenVLAOFTPolicy) -> None:
    obs = _obs(3, seed=3)
    policy.eval()
    with torch.no_grad():
        batch = policy.action_logits(policy.inputs(obs, INSTR))
        for i in range(3):
            single = {k: v[i : i + 1] for k, v in obs.items()}
            one = policy.action_logits(policy.inputs(single, INSTR[i : i + 1]))
            assert torch.allclose(batch[i : i + 1], one, atol=1e-5), i


def test_rl_contract_sample_logprob_sft_consistent(policy: OpenVLAOFTPolicy) -> None:
    assert isinstance(policy, VLAPolicy)
    obs = _obs(3, seed=4)
    policy.eval()
    torch.manual_seed(0)
    s = policy.sample(obs, INSTR, temperature=1.6)
    assert s.actions.shape == s.tokens.shape == s.log_probs.shape == (3, 8, 6)
    assert s.tokens.dtype == torch.int64 and int(s.tokens.min()) >= 0 and int(s.tokens.max()) < 256
    for mode in (False, True):
        policy.train(mode)
        out = policy.log_prob(obs, INSTR, s.tokens, temperature=1.6)
        assert torch.allclose(out.log_probs, s.log_probs, atol=1e-5)
        assert out.entropy.shape == (3, 8, 6)
    out.log_probs.sum().backward()
    assert any(p.grad is not None for p in policy.trainable_parameters())
    policy.zero_grad(set_to_none=True)
    # greedy == act_batch, and decoded actions stay inside the bounds
    policy.eval()
    g = policy.sample(obs, INSTR, temperature=0.0)
    assert torch.allclose(g.actions, policy.act_batch(obs, INSTR))
    lo, hi = torch.tensor(policy.pre.action_bounds.low), torch.tensor(policy.pre.action_bounds.high)
    assert bool(((g.actions >= lo - 1e-4) & (g.actions <= hi + 1e-4)).all())
    single = policy.act(
        {"image": obs["image"][0].numpy(), "state": obs["state"][0].numpy()}, INSTR[0]
    )
    assert single.shape == (8, 6) and np.allclose(single, g.actions[0].numpy(), atol=1e-5)
    # the SFT forward scores the same logits the sampler uses (deterministic crop when train=False)
    tok = policy.pre.build_inputs(INSTR, obs["image"], obs["state"], g.actions).targets
    assert tok is not None and torch.equal(tok.view(3, 8, 6), g.tokens)


def test_sft_step_learns_and_grad_ckpt_matches(cfg: dict[str, Any]) -> None:
    batch = SFTBatch(
        images=torch.rand(2, 3, 96, 96),
        states=torch.zeros(2, 6),
        actions=torch.rand(2, 8, 6) * 0.5,
        instructions=INSTR[:2],
    )
    grads = []
    for ckpt in (False, True):
        pol = build_policy(cfg, "cpu", grad_checkpointing=ckpt)
        pol.train()
        pol._aug_gen.manual_seed(0)
        loss, metrics = pol.sft_loss(batch)
        loss.backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in pol.trainable_parameters() if p.grad is not None])
        )
        assert set(metrics) == {"loss", "token_acc", "l1_norm"}
    assert torch.allclose(grads[0], grads[1], atol=1e-5)
    # a few steps on one batch reduce the loss
    pol = build_policy(cfg, "cpu")
    pol.train()
    opt = torch.optim.AdamW(pol.trainable_parameters(), lr=3e-3)
    first = last = 0.0
    for i in range(15):
        pol._aug_gen.manual_seed(0)
        loss, _ = pol.sft_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if i == 0 else first
        last = float(loss)
    assert last < first


def test_save_load_roundtrip(cfg: dict[str, Any], tmp_path: Path) -> None:
    pol = build_policy(cfg, "cpu")
    with torch.no_grad():
        for p in pol.trainable_parameters():
            p.add_(0.01 * torch.randn_like(p))
    pol.save(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "norm_stats.json").is_file()
    again = build_policy(cfg, "cpu", adapter=tmp_path / "adapter")
    obs = _obs(2, seed=5)
    pol.eval()
    again.eval()
    with torch.no_grad():
        a = pol.action_logits(pol.inputs(obs, INSTR[:2]))
        b = again.action_logits(again.inputs(obs, INSTR[:2]))
    assert torch.allclose(a, b, atol=1e-6)
    assert again.pre.action_bounds == pol.pre.action_bounds
    rl = build_policy(cfg, "cpu", adapter=tmp_path / "adapter", lora_only=True)
    names = [n for n, p in rl.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
