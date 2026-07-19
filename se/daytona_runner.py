"""Real SWE-bench evaluation inside a Daytona sandbox — the piece that replaces
the modeled scores in fixers.py with ground-truth pass/fail.

Architecture (path 1 from the research): create a Daytona sandbox directly FROM
the official SWE-bench per-instance image (which already has the repo checked out
at the base commit with the exact dependencies), apply a candidate patch, run the
official eval script, and grade with SWE-bench's own grader — all *inside* the
Linux sandbox. The local (Windows) side only drives Daytona; it never imports
swebench (which needs the Unix-only `resource` module).

Resource note: Daytona caps a sandbox at 4 vCPU / 8 GB / 10 GB disk. The instance
image is the sandbox rootfs (no nested Docker), so light repos (flask, requests,
pytest, sqlfluff, marshmallow) fit comfortably; heavy scientific repos may not.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field

from .swebench_data import SweInstance

DEFAULT_NAMESPACE = "swebench"   # Docker Hub org that hosts the eval images
DEFAULT_IMAGE_TAG = "latest"


def instance_image(inst: SweInstance, namespace: str = DEFAULT_NAMESPACE,
                   tag: str = DEFAULT_IMAGE_TAG) -> str:
    """Reproduce swebench's `TestSpec.instance_image_key` for the remote image
    (namespace set) without importing swebench: lowercase id, `__` -> `_1776_`."""
    key = f"sweb.eval.x86_64.{inst.instance_id.lower()}:{tag}"
    return f"{namespace}/{key}".replace("__", "_1776_")


# The driver runs INSIDE the sandbox (Linux, where swebench imports fine). It
# applies the model patch to /testbed, runs the official eval script, and grades
# with swebench's own get_eval_report, printing a single RESULT_JSON line.
_EVAL_DRIVER = r'''
import json, subprocess, sys
try:
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_report
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION
except Exception as e:
    print("RESULT_JSON:" + json.dumps({"error": "swebench import failed: %r" % e}))
    sys.exit(0)

inst = json.load(open("/tmp/instance.json"))
patch = open("/tmp/model.patch").read()
iid = inst["instance_id"]
spec = make_test_spec(inst)

applied = False
if patch.strip():
    open("/tmp/apply.patch", "w").write(patch)
    for cmd in ("git apply --verbose /tmp/apply.patch",
                "git apply --verbose --reject /tmp/apply.patch",
                "patch --batch --fuzz=5 -p1 -i /tmp/apply.patch"):
        r = subprocess.run(cmd, shell=True, cwd="/testbed",
                           capture_output=True, text=True)
        if r.returncode == 0:
            applied = True
            break

eval_script = spec.eval_script

# Optional test-order perturbation: pytest executes node IDs in the order given
# on the command line, so we rewrite the test command between the output markers
# with an explicitly shuffled node-ID list. The eval script resets and re-applies
# the test patch itself, so editing the test FILE would not survive -- this is the
# only place the order can actually be changed.
try:
    order = json.load(open("/tmp/test_order.json"))
except Exception:
    order = None
if order:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    tc = MAP_REPO_VERSION_TO_SPECS[inst["repo"]][inst["version"]]["test_cmd"]
    if isinstance(tc, list):
        tc = tc[-1]
    quoted = " ".join("'%s'" % n.replace("'", "'\\''") for n in order)
    rewritten, in_test = [], False
    for line in eval_script.split("\n"):
        if "START_TEST_OUTPUT" in line:
            in_test = True
            rewritten.append(line)
            continue
        if "END_TEST_OUTPUT" in line:
            in_test = False
            rewritten.append(line)
            continue
        if in_test and line.strip():
            rewritten.append("%s %s" % (tc, quoted))   # replace directives
        else:
            rewritten.append(line)
    eval_script = "\n".join(rewritten)

open("/tmp/eval.sh", "w").write(eval_script)
with open("/tmp/test_output.txt", "w") as out:
    subprocess.run("bash /tmp/eval.sh", shell=True, stdout=out, stderr=subprocess.STDOUT)

pred = {KEY_INSTANCE_ID: iid, KEY_PREDICTION: patch, "model_name_or_path": "daytona"}
report = get_eval_report(spec, pred, "/tmp/test_output.txt", include_tests_status=True)[iid]
ts = report.get("tests_status", {}) or {}

def counts(k):
    d = ts.get(k, {}) or {}
    return len(d.get("success", [])), len(d.get("failure", []))

f2p_s, f2p_f = counts("FAIL_TO_PASS")
p2p_s, p2p_f = counts("PASS_TO_PASS")
passed, total = f2p_s + p2p_s, f2p_s + f2p_f + p2p_s + p2p_f
print("RESULT_JSON:" + json.dumps({
    "instance_id": iid,
    "resolved": bool(report.get("resolved", False)),
    "patch_applied": bool(report.get("patch_successfully_applied", False)) and applied,
    "f2p_success": f2p_s, "f2p_fail": f2p_f,
    "p2p_success": p2p_s, "p2p_fail": p2p_f,
    "passed": passed, "total": total,
    "score": (100.0 * passed / total) if total else 0.0,
}))
'''


@dataclass
class EvalResult:
    instance_id: str
    resolved: bool
    patch_applied: bool
    passed: int
    total: int
    score: float               # 100 * passed / total
    f2p_success: int = 0
    f2p_fail: int = 0
    p2p_success: int = 0
    p2p_fail: int = 0
    raw: dict = field(default_factory=dict)
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return f"{self.instance_id}: ERROR {self.error}"
        verdict = "RESOLVED" if self.resolved else "not resolved"
        return (f"{self.instance_id}: {verdict}  score={self.score:.1f}  "
                f"(F2P {self.f2p_success}/{self.f2p_success + self.f2p_fail}, "
                f"P2P {self.p2p_success}/{self.p2p_success + self.p2p_fail}, "
                f"patch_applied={self.patch_applied})")


class DaytonaRunner:
    """Evaluate patches on real SWE-bench instances via Daytona sandboxes."""

    def __init__(self, cpu: int = 2, memory: int = 4, disk: int = 10,
                 namespace: str = DEFAULT_NAMESPACE,
                 python_bin: str = "/opt/miniconda3/bin/python",
                 pip_bin: str = "/opt/miniconda3/bin/pip",
                 auto_stop_min: int = 15, verbose: bool = True):
        from daytona import Daytona, DaytonaConfig

        from .secrets import require_key
        # Key comes from the environment or the gitignored project .env — it is
        # passed straight to the client and never printed.
        self.daytona = Daytona(DaytonaConfig(api_key=require_key("DAYTONA_API_KEY")))
        self.cpu, self.memory, self.disk = cpu, memory, disk
        self.namespace = namespace
        self.python_bin, self.pip_bin = python_bin, pip_bin
        self.auto_stop_min = auto_stop_min
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print("[daytona]", *a, flush=True)

    @staticmethod
    def _upload_text(sb, path: str, text: str) -> None:
        """Write text into the sandbox via chunked base64 (SDK-version-agnostic)."""
        sb.process.exec(f"mkdir -p $(dirname {path}) && : > {path}")
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        for i in range(0, len(b64), 48000):  # stay well under arg-length limits
            chunk = b64[i:i + 48000]
            sb.process.exec(f"printf %s '{chunk}' | base64 -d >> {path}")

    def evaluate_patch(self, inst: SweInstance, model_patch: str,
                       timeout: int = 1800,
                       file_overrides: dict[str, str] | None = None,
                       new_files: dict[str, str] | None = None,
                       test_order: list[str] | None = None) -> EvalResult:
        """Create a sandbox from the instance image, apply `model_patch`, run the
        official eval, and return the real pass/fail. An empty patch measures the
        pre-fix baseline (should NOT resolve).

        `file_overrides` / `new_files` apply a repo-side perturbation before the
        patch is applied — used to verify that a perturbation is genuinely
        meaning-preserving (grade the GOLD patch under it; it must still resolve)."""
        from daytona import CreateSandboxFromImageParams, Resources
        image = instance_image(inst, self.namespace)
        self._log(f"creating sandbox from {image}")
        sb = self.daytona.create(
            CreateSandboxFromImageParams(
                image=image,
                resources=Resources(cpu=self.cpu, memory=self.memory, disk=self.disk),
                auto_stop_interval=self.auto_stop_min,
                ephemeral=True,
            ),
            timeout=300,
        )
        try:
            # Repo-side perturbation goes in FIRST, so the patch and the eval both
            # run against the perturbed tree.
            for path, content in (file_overrides or {}).items():
                self._log(f"perturbing repo file {path}")
                self._upload_text(sb, f"/testbed/{path}", content)
            for path, content in (new_files or {}).items():
                self._log(f"adding repo file {path}")
                self._upload_text(sb, f"/testbed/{path}", content)

            self._upload_text(sb, "/tmp/instance.json", json.dumps(inst.raw))
            self._upload_text(sb, "/tmp/model.patch", model_patch or "")
            self._upload_text(sb, "/tmp/driver.py", _EVAL_DRIVER)
            if test_order:
                self._log(f"shuffling execution order of {len(test_order)} tests")
                self._upload_text(sb, "/tmp/test_order.json", json.dumps(test_order))

            self._log("installing swebench in sandbox")
            r = sb.process.exec(f"{self.pip_bin} install --quiet swebench 2>&1 | tail -3",
                                timeout=600)
            self._log(f"running eval (up to {timeout}s)")
            resp = sb.process.exec(f"{self.python_bin} /tmp/driver.py", timeout=timeout)
            out = resp.result or ""
            m = re.search(r"RESULT_JSON:(\{.*\})", out)
            if not m:
                return EvalResult(inst.instance_id, False, False, 0, 0, 0.0,
                                  error="no RESULT_JSON in driver output",
                                  raw={"stdout_tail": out[-1500:]})
            d = json.loads(m.group(1))
            if "error" in d:
                return EvalResult(inst.instance_id, False, False, 0, 0, 0.0, error=d["error"])
            return EvalResult(
                instance_id=d["instance_id"], resolved=d["resolved"],
                patch_applied=d["patch_applied"], passed=d["passed"], total=d["total"],
                score=d["score"], f2p_success=d["f2p_success"], f2p_fail=d["f2p_fail"],
                p2p_success=d["p2p_success"], p2p_fail=d["p2p_fail"], raw=d)
        finally:
            try:
                sb.delete()
                self._log("sandbox deleted")
            except Exception as e:  # pragma: no cover
                self._log(f"warning: sandbox cleanup failed: {e!r}")

    def smoke_test(self, inst: SweInstance) -> tuple[EvalResult, EvalResult]:
        """Zero-API-cost pipeline check: the gold patch must resolve, an empty
        patch must not. If both hold, the Daytona + SWE-bench path is correct."""
        self._log(f"SMOKE TEST on {inst.instance_id}")
        gold = self.evaluate_patch(inst, inst.patch)
        self._log("gold  -> " + gold.summary())
        empty = self.evaluate_patch(inst, "")
        self._log("empty -> " + empty.summary())
        ok = gold.resolved and not empty.resolved
        self._log(f"SMOKE TEST {'PASSED' if ok else 'FAILED'} "
                  f"(gold resolved={gold.resolved}, empty resolved={empty.resolved})")
        return gold, empty
