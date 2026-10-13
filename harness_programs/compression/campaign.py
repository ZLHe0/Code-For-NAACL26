"""Stage 1 of Algorithm 1 (lines 1-3): Rollout -> ProposeCompress -> Gate, emitting H0.

Every phase writes a JSON artifact under the output directory and is skipped on a re-run when
its artifact already exists, so an interrupted campaign resumes where it stopped. Delete an
artifact to re-run its phase and everything after it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import splits
from ..domains import load as load_domain
from ..domains.base import Task
from ..llm.client import ChatModel
from . import emit, gate, propose, refine, rollout
from .episodes import OptEpisode, Sizing, Target, load_episodes

logger = logging.getLogger(__name__)

PHASES = ("rollout", "propose", "gate")


@dataclass
class CampaignConfig:
    """Everything a campaign needs.

    Args:
        domain: registered domain name.
        out_dir: where artifacts and the final ``program.json`` are written.
        sizing: the context limit training serves under.
        tasks_path: the training tasks D_train the policy trains on, for the rollout.
        holdout_path: the held-out *training* tasks D_hold, for the gate: a slice carved out of
            the training split (:mod:`harness_programs.splits`), disjoint from D_train and from every
            evaluation set. Never the validation file; the campaign refuses to start otherwise.
        eval_tasks: evaluation task files that D_hold is checked to be disjoint from.
        episodes: pre-collected rollouts (JSONL) to use as R0 instead of rolling out.
        rollout_tasks: training tasks rolled out.
        group_size: rollouts per training task.
        candidates: candidate programs the creator drafts.
        rounds: accept-or-revise rounds per candidate.
        holdout_tasks: D_hold tasks each program is scored on.
    """

    domain: str
    out_dir: str
    sizing: Sizing
    tasks_path: str = ""
    holdout_path: str = ""
    eval_tasks: tuple[str, ...] = ()
    episodes: tuple[str, ...] = ()
    rollout_tasks: int = 64
    group_size: int = rollout.DEFAULT_GROUP_SIZE
    candidates: int = propose.DEFAULT_CANDIDATES
    rounds: int = refine.DEFAULT_ROUNDS
    holdout_tasks: int = 64
    workers: int = 8


class Campaign:
    def __init__(self, config: CampaignConfig, *, llm: ChatModel, policy: ChatModel) -> None:
        self.cfg = config
        self.llm = llm
        self.policy = policy
        self.out = Path(config.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.domain = load_domain(config.domain)
        splits.check(self.domain, holdout=config.holdout_path, train=config.tasks_path,
                     evaluation=config.eval_tasks)
        self.target = Target(config.domain, config.sizing)

    def run(self) -> dict[str, Any]:
        reservoir = self.rollout()
        proposals = self.propose(reservoir)
        return self.gate(proposals)

    # -- phases ----------------------------------------------------------------------------

    def rollout(self) -> list[OptEpisode]:
        """R0: the base policy on D_train, append-only, ``group_size`` rollouts per task."""
        info = self._artifact("rollout", self._collect)
        eps = load_episodes(info["episode_files"], harness=self.cfg.domain)
        if not eps:
            raise ValueError(f"no {self.cfg.domain} episodes in {info['episode_files']}")
        return eps

    def propose(self, reservoir: list[OptEpisode]) -> list[dict[str, Any]]:
        """P: the creator's accepted candidate programs."""
        result = self._artifact("propose", lambda: propose.run(self.llm, self.target, reservoir,
                                                               candidates=self.cfg.candidates, rounds=self.cfg.rounds))
        return result["accepted"]

    def gate(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        """H0: the best of P on D_hold if it beats no compression; written to ``program.json``."""
        result = self._artifact("gate", lambda: gate.run(
            self.domain, proposals, self._tasks(self.cfg.holdout_path, splits.HOLDOUT_SPLIT, self.cfg.holdout_tasks,
                                     "--holdout"),
            self.policy, self.cfg.sizing, self.out / "gate", workers=self.cfg.workers))
        program_path = self.out / "program.json"
        emit.run(self.cfg.domain, result["compression"], program_path)
        logger.info("H0 = %s -> %s", result["winner"], program_path)
        return {"winner": result["winner"], "scores": result["scores"], "program": str(program_path)}

    # -- helpers ---------------------------------------------------------------------------

    def _artifact(self, phase: str, build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.out / f"{phase}.json"
        if path.exists():
            logger.info("%s: resuming from %s", phase, path)
            return json.loads(path.read_text())
        result = build()
        path.write_text(json.dumps(result, indent=2, default=str))
        return result

    def _tasks(self, path: str, split: str, limit: int, flag: str) -> list[Task]:
        if not path:
            raise ValueError(f"{self.cfg.domain}: this phase needs {flag}")
        return self.domain.load_tasks(path, split=split, limit=limit)

    def _collect(self) -> dict[str, Any]:
        if self.cfg.episodes:
            files: list[str | Path] = list(self.cfg.episodes)
            eps = load_episodes(files, harness=self.cfg.domain)
        else:
            path = self.out / "rollout" / "episodes.jsonl"
            files = [path]
            if path.exists():
                eps = load_episodes(path, harness=self.cfg.domain)
            else:
                tasks = self._tasks(self.cfg.tasks_path, "train", self.cfg.rollout_tasks, "--tasks or --episodes")
                eps = rollout.run(self.domain, tasks, self.policy, self.cfg.sizing, path,
                                  group_size=self.cfg.group_size, workers=self.cfg.workers)
        return rollout.summary(eps, files)
