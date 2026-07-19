"""DaytonaFixer — the real fixer that replaces fixers.py's modeled scores.

It satisfies the same interface as the modeled `Fixer` (a `.name` and
`score_under(inst, perturbation_kind, rng) -> float`), so it drops straight into
`evaluate_skeptically` / `evaluate_adaptive`. Under the hood each call:

  1. builds the perturbed input for that channel via the Perturbation Agent,
  2. runs the real code-fix agent on it (in a Daytona sandbox) to get a patch,
  3. grades that patch on a fresh sandbox with the official SWE-bench eval,

and returns the real % of tests passing. Patches are cached per distinct input so
we never pay for the same agent run twice.

COST: each distinct channel = one agent run + one eval = ~2 Daytona sandboxes plus
one Anthropic agent loop. A full `evaluate_skeptically` over ~10 safe sites is
~20 sandboxes + ~10 agent runs per candidate. Prefer `evaluate_adaptive` (probes
one perturbation per axis, stops early) and a light-repo instance to keep spend
bounded. Run scripts/07 (smoke test) first — it costs no Anthropic tokens.
"""

from __future__ import annotations

from .agent import SimpleSweAgent
from .daytona_runner import DaytonaRunner
from .patch_inspect import Finding, inspect_patch, summarize
from .perturbations import PerturbationAgent
from .swebench_data import SweInstance


class DaytonaFixer:
    name = "daytona-real"

    def __init__(self, runner: DaytonaRunner | None = None,
                 agent: SimpleSweAgent | None = None,
                 perturbation_agent: PerturbationAgent | None = None,
                 verbose: bool = True):
        self.runner = runner or DaytonaRunner(verbose=verbose)
        self.agent = agent or SimpleSweAgent(verbose=verbose)
        self.pert = perturbation_agent or PerturbationAgent()
        self.verbose = verbose
        self._sites: dict[str, dict] = {}          # instance_id -> {kind: site}
        self._patch_cache: dict[tuple, str] = {}    # input signature -> patch
        self._score_cache: dict[tuple, float] = {}  # (instance_id, kind) -> score
        self.inspections: dict[tuple, list[Finding]] = {}  # (instance_id, kind) -> findings

    def _log(self, *a):
        if self.verbose:
            print("[fixer]", *a, flush=True)

    def _sites_for(self, inst: SweInstance) -> dict:
        if inst.instance_id not in self._sites:
            self._sites[inst.instance_id] = {
                s.kind: s for s in self.pert.find_sites(inst) if s.safe}
        return self._sites[inst.instance_id]

    def _patch_for_input(self, inst, problem, overrides, new_files) -> str:
        sig = (inst.instance_id, problem,
               tuple(sorted((overrides or {}).items())),
               tuple(sorted((new_files or {}).items())))
        if sig not in self._patch_cache:
            self._patch_cache[sig] = self.agent.generate_patch(
                self.runner, inst, problem_statement=problem,
                file_overrides=overrides, new_files=new_files)
        return self._patch_cache[sig]

    def score_under(self, inst: SweInstance, perturbation_kind: str, rng=None) -> float:
        cache_key = (inst.instance_id, perturbation_kind)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if perturbation_kind == "canonical":
            patch = self._patch_for_input(inst, inst.problem_statement, None, None)
        else:
            site = self._sites_for(inst).get(perturbation_kind)
            if site is None:
                raise ValueError(f"no safe site for kind {perturbation_kind!r} on {inst.instance_id}")
            ap = self.pert.apply(inst, site)
            if site.side == "evaluator":
                # e.g. test_reorder: the fixer input is unchanged, so reuse the
                # canonical patch. (A true test-order reshuffle in the harness is
                # a documented next step; this still measures eval-side stability.)
                patch = self._patch_for_input(inst, inst.problem_statement, None, None)
            else:
                patch = self._patch_for_input(inst, ap.problem_statement,
                                              ap.file_overrides, ap.new_files)

        # Static cheat detection runs on every real patch — free, and it catches the
        # tampering class that perturbation variance is blind to.
        findings = inspect_patch(patch)
        self.inspections[cache_key] = findings
        if findings:
            self._log(f"{perturbation_kind}: patch inspection -> {summarize(findings)}")
            for f in findings:
                self._log(f"    {f}")

        self._log(f"{perturbation_kind}: grading patch ({len(patch)} chars)")
        result = self.runner.evaluate_patch(inst, patch)
        self._log(f"{perturbation_kind}: {result.summary()}")
        self._score_cache[cache_key] = result.score
        return result.score
