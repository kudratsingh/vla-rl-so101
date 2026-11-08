"""Spend gates on windowed rollout success and entropy (review 3.1 / 5; replaces the WO-04 kill
switch and peak stop).

WHAT THESE GATES ARE FOR: deciding whether to keep paying for a run. They are NOT a measurement of
improvement. Rollout episodes come from a policy that changes every step, share scenes within a
group, and are drawn from the training prompt range; only the fixed held-out protocol eval
(src/eval/protocol.py, scripts/wo02_eval.py) decides whether the policy improved. The periodic
greedy evals in the trainer are sanity logs only.

Why rollout success instead of the old greedy evals: the WO-04 kill switch compared two 64-episode
greedy evals (paired-difference SD 4-6 points at p ~0.13 against a +3 bar) and the peak stop two
32-episode evals; neither could tell +3 from 0. `rollout/success_rate` is free and covers every
generated episode (~130-380 per step). The gates' unit is the GROUP, not the episode (a group's K
samples share one scene). configs/rl.yaml generates 8-24 groups per trained task per step (env
batches of 4 groups alternate between the 2 tasks; the dynamic-sampling refill stops at 16 kept
groups or 12 env batches). SIMULATED under the test noise model (scene p ~ Beta, concentration 2,
K = 8, that refill; script in the PR #28 body), at p 0.08 / 0.13 / 0.20 the refill gives ~24 / 21 /
16 groups per task per step, and:
  one 10-step window, per task          SE 1.1 / 1.5 / 2.1 points
  difference of two 10-step windows     SE 1.6 / 2.1 / 2.9 points
  flat-gate trend gain (38 steps)       SE 2.0 / 2.6 / 3.6 points (the SE the gate logs matches
                                        the simulated spread to within 2%)

Per step the trainer records, for each trained task, the number of generated groups and the first
two moments of the group-mean success (a group is the unit that is independent: its K samples
share one scene), plus the sampled-token entropy sum and count (an unbiased estimate of the
rollout policy's per-token entropy, src/rl/diagnostics.py). A window is the last `window` steps.

  per-task window rate  pooled group means of that task over the window; SE from their spread
  score                 equal-weight mean of the per-task window rates (robust to the step's task
                        mix, which the dynamic-sampling refill makes uneven)
  flat gate             at `check_step`, per task: the group-level least-squares trend of success
                        over EVERY recorded step 1..check_step; gain = its slope x (check_step - 1),
                        the fitted change from step 1 (the base policy: step k's rollouts come from
                        the policy after k - 1 updates) to the check step; gain_se from the group
                        residuals. Rule any_task: stop if, for every task, gain + flat_z x gain_se
                        < min_gain (mean: the task-mean gain and its SE). -> record status
                        stopped_flat (no protocol eval). min_gain is the owner's +3 points, now a
                        gain FROM THE SFT POLICY. Until 2026-11-08 the gain was window(29-38) -
                        window(1-10); the first window's policies had 0-9 updates (mean 4.5), so
                        early learning sat in the baseline (a linear learner's gain read ~25% low:
                        centres 28 steps apart, not 37). It is still logged as
                        gate/flat_window_gain. A linear trend reads a front-loaded learner low
                        too: +4 points by step 10 then flat reads +2.6 (+6 reads +3.85).
                        Operating characteristic, P(stop at step 38), SIMULATED (noise model
                        above, 2 tasks, 4000 runs per cell, p0 0.08 / 0.13 / 0.20, linear gain):
                          rule                          flat         one task +5  both tasks +5
                          older rollout-window rule     .94/.85/.72  .32/.32/.33  .12/.14/.17
                          trend, z 0                    .89/.78/.65  .17/.20/.23  .04/.07/.10
                          trend, z 0.5 (rl.yaml)        .73/.57/.40  .06/.08/.09  .01/.01/.02
                          trend, z 1                    .50/.33/.19  .02/.02/.03  .00/.00/.00
                        "older rollout-window rule" = window(29-38) - window(1-10) of sampled
                        rollout success, z 0 (this PR's first version). The table does NOT
                        reproduce the original 64-episode greedy gate (the WO-04 kill switch).
                        z 0.5 is the smallest of 0 / 0.3 / 0.4 / 0.5 / 1 with P(stop | one task
                        +5) <= 0.10 at every p0 (0.4: 0.101 / 0.106 at p0 0.13 / 0.20).
                        tests/test_rl_gates.py pins the p0 0.13 cells. Owner 2026-11-08: keep z 0.5 as the
                        provisional default (simulated only; revisit after a GPU run).
  decline gate          the best window is tracked over ALIGNED windows only (ending at
                        first_step - 1 + k x window, so not the running max of overlapping
                        windows). Stop when, for BOTH the window ending at this step and the one
                        ending `window` steps earlier (two consecutive non-overlapping windows,
                        neither overlapping the best), best - score > max(max_drop,
                        decline_z x sqrt(se_best^2 + se_window^2)). A sampling-noise dip does not
                        fire it. z = 3, not 2: the best is the max of up to steps/window windows,
                        so a z = 2 margin stopped 35/100 simulated FLAT runs (p 0.2, Beta
                        concentration 2, 8 groups/task/step, K = 8, 150 steps); z = 3 stopped
                        1/100 and still caught a 30% -> 10% decline in 100/100 (median step
                        58-83). tests/test_rl_gates.py pins the flat case. max_drop 0 = off.
  entropy gate          stop if the window's rollout entropy < entropy_min nats/token (sampling
                        has collapsed onto the argmax: groups go all-equal and are dropped, so
                        rollouts are paid for no gradient). 0 = off.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

GateReason = Literal["flat", "decline", "entropy"]


@dataclass(frozen=True)
class TaskTally:
    """One step's rollout outcome for one task: groups and the moments of their mean success."""

    groups: int
    sum_rate: float  # sum over groups of the group's mean success
    sum_rate_sq: float  # sum over groups of its square

    @staticmethod
    def from_group_rates(rates: Sequence[float]) -> TaskTally:
        return TaskTally(len(rates), float(sum(rates)), float(sum(r * r for r in rates)))


@dataclass(frozen=True)
class WindowRate:
    rate: float
    se: float  # standard error over groups (0 with a single group)
    groups: int


def pooled_rate(tallies: Sequence[TaskTally]) -> WindowRate | None:
    n = sum(t.groups for t in tallies)
    if n == 0:
        return None
    s = sum(t.sum_rate for t in tallies)
    s2 = sum(t.sum_rate_sq for t in tallies)
    rate = s / n
    var = max(0.0, (s2 / n - rate * rate) * n / (n - 1)) if n > 1 else 0.0
    return WindowRate(rate=rate, se=math.sqrt(var / n), groups=n)


@dataclass
class StepRecord:
    step: int
    tasks: dict[str, TaskTally]
    entropy_sum: float
    entropy_tokens: int

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tasks": {k: [v.groups, v.sum_rate, v.sum_rate_sq] for k, v in self.tasks.items()},
            "entropy_sum": self.entropy_sum,
            "entropy_tokens": self.entropy_tokens,
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> StepRecord:
        tasks = {k: TaskTally(int(v[0]), float(v[1]), float(v[2])) for k, v in d["tasks"].items()}
        return StepRecord(int(d["step"]), tasks, float(d["entropy_sum"]), int(d["entropy_tokens"]))


@dataclass
class GateResult:
    metrics: dict[str, float] = field(default_factory=dict)
    reason: GateReason | None = None  # the gate that fired (None: keep spending)


class RolloutGates:
    def __init__(
        self,
        *,
        tasks: Sequence[str],
        window: int,
        check_step: int,
        min_gain: float,
        rule: Literal["any_task", "mean"],
        max_drop: float,
        entropy_min: float,
        decline_z: float = 3.0,
        flat_z: float = 0.0,
    ) -> None:
        if window < 1:
            raise ValueError("gate window must be >= 1 step")
        if check_step and check_step < 2 * window:
            raise ValueError(
                f"flat-gate check step {check_step} < 2 x gate_window {window}: the first window "
                "and the check window would overlap"
            )
        if decline_z < 0:
            raise ValueError("decline_z must be >= 0")
        if flat_z < 0:
            raise ValueError("flat_z must be >= 0")
        self.tasks = list(tasks)
        self.window = window
        self.check_step = check_step
        self.min_gain = min_gain
        self.rule = rule
        self.max_drop = max_drop
        self.entropy_min = entropy_min
        self.decline_z = decline_z
        self.flat_z = flat_z
        self.history: list[StepRecord] = []
        self.best_score: float | None = None
        self.best_se = 0.0
        self.best_step = 0

    # ------------------------------------------------------------------ state (resume)
    def state_dict(self) -> dict[str, Any]:
        return {
            "history": [r.to_json() for r in self.history],
            "best_score": self.best_score,
            "best_se": self.best_se,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.history = [StepRecord.from_json(r) for r in state["history"]]
        best = state["best_score"]
        self.best_score = None if best is None else float(best)
        self.best_se = float(state.get("best_se", 0.0))
        self.best_step = int(state["best_step"])

    def truncate(self, last_step: int) -> None:
        """Forget records after `last_step` (approximate resume from an _interrupted checkpoint)
        and recompute the best aligned window from what is left."""
        if self.best_step <= last_step and all(r.step <= last_step for r in self.history):
            return
        self.history = [r for r in self.history if r.step <= last_step]
        self.best_score, self.best_se, self.best_step = None, 0.0, 0
        for r in self.history:
            if not (self._full(r.step) and self._aligned(r.step)):
                continue
            scored = self._score(self.task_rates(r.step))
            if scored is not None and (self.best_score is None or scored[0] > self.best_score):
                self.best_score, self.best_se, self.best_step = scored[0], scored[1], r.step

    # ------------------------------------------------------------------ windows
    def record(
        self,
        step: int,
        tasks: Mapping[str, TaskTally],
        entropy_sum: float,
        entropy_tokens: int,
    ) -> None:
        if self.history and step <= self.history[-1].step:
            raise ValueError(f"gate steps must increase ({step} after {self.history[-1].step})")
        self.history.append(StepRecord(step, dict(tasks), entropy_sum, entropy_tokens))

    def _window(self, last_step: int) -> list[StepRecord]:
        return [r for r in self.history if last_step - self.window < r.step <= last_step]

    def task_rates(self, last_step: int) -> dict[str, WindowRate]:
        recs = self._window(last_step)
        out = {}
        for t in self.tasks:
            wr = pooled_rate([r.tasks[t] for r in recs if t in r.tasks])
            if wr is not None:
                out[t] = wr
        return out

    @staticmethod
    def _score(rates: Mapping[str, WindowRate]) -> tuple[float, float] | None:
        if not rates:
            return None
        n = len(rates)
        score = sum(r.rate for r in rates.values()) / n
        se = math.sqrt(sum(r.se**2 for r in rates.values())) / n
        return score, se

    def entropy(self, last_step: int) -> float | None:
        recs = self._window(last_step)
        tokens = sum(r.entropy_tokens for r in recs)
        return sum(r.entropy_sum for r in recs) / tokens if tokens else None

    def _full(self, last_step: int) -> bool:
        return len(self._window(last_step)) == self.window

    def _aligned(self, last_step: int) -> bool:
        """Window ending at `last_step` is one of the non-overlapping tiles from the first step."""
        return bool(self.history) and (last_step - self.history[0].step + 1) % self.window == 0

    def _declined(self, last_step: int) -> tuple[float, float] | None:
        """(drop, margin) of the window ending at `last_step` below the best window, or None
        when that window is not full or overlaps the best."""
        if self.best_score is None or last_step - self.best_step < self.window:
            return None
        if not self._full(last_step):
            return None
        scored = self._score(self.task_rates(last_step))
        if scored is None:
            return None
        score, se = scored
        margin = max(self.max_drop, self.decline_z * math.sqrt(self.best_se**2 + se**2))
        return self.best_score - score, margin

    # ------------------------------------------------------------------ gates
    def evaluate(self, step: int) -> GateResult:
        """Window metrics after `step` (already recorded) and the gate that fires, if any."""
        out = GateResult()
        if not self._full(step):
            return out
        m = out.metrics
        rates = self.task_rates(step)
        for t, wr in rates.items():
            m[f"gate/success_window/{t}"] = wr.rate
            m[f"gate/success_window_se/{t}"] = wr.se
        scored = self._score(rates)
        ent = self.entropy(step)
        if ent is not None:
            m["gate/entropy_window"] = ent
        if scored is not None:
            score, se = scored
            m["gate/success_window"] = score
            m["gate/success_window_se"] = se
            if self._aligned(step) and (self.best_score is None or score > self.best_score):
                self.best_score, self.best_se, self.best_step = score, se, step
            if self.best_score is not None:
                m["gate/best_success_window"] = self.best_score
                m["gate/best_success_window_se"] = self.best_se
                m["gate/best_success_step"] = float(self.best_step)

        reasons: list[GateReason] = []
        if self.check_step and step == self.check_step:
            gain, gain_se = self._flat_gain(step)
            upper = gain + self.flat_z * gain_se
            m["gate/flat_gain"], m["gate/flat_gain_se"] = gain, gain_se
            m["gate/flat_gain_upper"] = upper
            for t, (g_t, se_t) in self._trend_gains(step).items():
                m[f"gate/flat_gain/{t}"], m[f"gate/flat_gain_se/{t}"] = g_t, se_t
            window_gain = self._window_gain(step)
            if window_gain is not None:  # the pre-2026-11-08 estimator, logged for comparison
                m["gate/flat_window_gain"] = window_gain
            if upper < self.min_gain:
                m["event/flat_run"] = 1.0
                reasons.append("flat")
        now = self._declined(step) if self.max_drop > 0 else None
        if now is not None:
            m["gate/decline_drop"], m["gate/decline_margin"] = now
            before = self._declined(step - self.window)
            if now[0] > now[1] and before is not None and before[0] > before[1]:
                m["event/success_decline"] = 1.0
                reasons.append("decline")
        if self.entropy_min > 0 and ent is not None and ent < self.entropy_min:
            m["event/entropy_collapse"] = 1.0
            reasons.append("entropy")
        out.reason = reasons[0] if reasons else None
        return out

    def _trend_gains(self, step: int) -> dict[str, tuple[float, float]]:
        """Per task: (gain, SE) of the group-level least-squares trend of success over every
        recorded step up to `step`, as the fitted change from the first recorded step (the base
        policy: its rollouts precede any update) to `step`. Tasks with < 3 groups or one step
        are left out."""
        first = self.history[0].step
        span = step - first
        out: dict[str, tuple[float, float]] = {}
        for t in self.tasks:
            pts = [(r.step, r.tasks[t]) for r in self.history if r.step <= step and t in r.tasks]
            pts = [(x, tt) for x, tt in pts if tt.groups > 0]
            n = sum(tt.groups for _, tt in pts)
            if n < 3 or len({x for x, _ in pts}) < 2:
                continue
            xbar = sum(x * tt.groups for x, tt in pts) / n
            ybar = sum(tt.sum_rate for _, tt in pts) / n
            sxx = sum(tt.groups * (x - xbar) ** 2 for x, tt in pts)
            sxy = sum((x - xbar) * tt.sum_rate for x, tt in pts)
            slope = sxy / sxx
            tss = sum(tt.sum_rate_sq for _, tt in pts) - n * ybar * ybar
            rss = max(0.0, tss - slope * sxy)
            se = math.sqrt(rss / (n - 2) / sxx)
            out[t] = (slope * span, se * span)
        return out

    def _flat_gain(self, step: int) -> tuple[float, float]:
        """(gain, its SE) at `step` from the per-task trends. any_task: the task with the
        highest gain + flat_z x SE; mean: the task mean. No task -> gain -inf (flat)."""
        trends = self._trend_gains(step)
        if not trends:
            return -math.inf, 0.0
        if self.rule == "any_task":
            best = max(trends, key=lambda t: trends[t][0] + self.flat_z * trends[t][1])
            return trends[best]
        n = len(trends)
        gain = sum(g for g, _ in trends.values()) / n
        return gain, math.sqrt(sum(se**2 for _, se in trends.values())) / n

    def _window_gain(self, step: int) -> float | None:
        """Old estimator (diagnostic only): window ending at `step` - first full window, best
        task (any_task) or task mean. Biased low: the first window's policies already had
        0..window-1 updates."""
        first_step = self.history[0].step + self.window - 1
        now, first = self.task_rates(step), self.task_rates(first_step)
        common = [t for t in self.tasks if t in now and t in first]
        if not common:
            return None
        gains = [now[t].rate - first[t].rate for t in common]
        return max(gains) if self.rule == "any_task" else sum(gains) / len(gains)
