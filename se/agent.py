"""A minimal but real SWE-bench code-fix agent.

It runs an Anthropic tool-use loop whose single tool — `run_bash` — executes in a
Daytona sandbox where the repo is checked out at /testbed. The agent explores and
edits the source there; when it stops, we capture `git diff` as the patch. That
patch is then graded on a *fresh* sandbox by daytona_runner (clean base + hidden
tests), so exploration and evaluation never contaminate each other.

Swap this out for Aider / SWE-agent later — DaytonaFixer only needs something with
`generate_patch(...) -> unified_diff_str`.
"""

from __future__ import annotations

import base64

from .secrets import require_key
from .swebench_data import SweInstance

DEFAULT_MODEL = "claude-opus-4-8"   # cost lever: pass model="claude-sonnet-5" for cheaper, high-volume runs

SYSTEM = (
    "You are a senior software engineer fixing ONE bug in a Python repository that "
    "is checked out at /testbed. Use the run_bash tool to explore the code and to "
    "edit files in place (sed, python, or writing files with heredocs). Make the "
    "smallest change that genuinely fixes the described bug. Do not edit tests. "
    "When the source is fixed, stop replying with tool calls and give a one-sentence "
    "summary of the fix — nothing else."
)

_TOOLS = [{
    "name": "run_bash",
    "description": "Run a bash command inside the repo working directory (/testbed). "
                   "Returns combined stdout/stderr (truncated).",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command."}},
        "required": ["command"],
    },
}]


def _upload_text(sb, path: str, text: str) -> None:
    sb.process.exec(f"mkdir -p $(dirname {path}) && : > {path}")
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    for i in range(0, len(b64), 48000):
        sb.process.exec(f"printf %s '{b64[i:i + 48000]}' | base64 -d >> {path}")


class SimpleSweAgent:
    def __init__(self, model: str = DEFAULT_MODEL, max_steps: int = 16,
                 max_tokens: int = 4096, verbose: bool = True):
        import anthropic
        self.client = anthropic.Anthropic(api_key=require_key("ANTHROPIC_API_KEY"))
        self.model, self.max_steps, self.max_tokens = model, max_steps, max_tokens
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            print("[agent]", *a, flush=True)

    def generate_patch(
        self,
        runner,                      # DaytonaRunner (for sandbox creation params)
        inst: SweInstance,
        problem_statement: str | None = None,
        file_overrides: dict[str, str] | None = None,
        new_files: dict[str, str] | None = None,
    ) -> str:
        """Explore + edit in a sandbox; return the resulting unified diff."""
        from daytona import CreateSandboxFromImageParams, Resources
        from .daytona_runner import instance_image

        problem = problem_statement if problem_statement is not None else inst.problem_statement
        sb = runner.daytona.create(
            CreateSandboxFromImageParams(
                image=instance_image(inst, runner.namespace),
                resources=Resources(cpu=runner.cpu, memory=runner.memory, disk=runner.disk),
                auto_stop_interval=runner.auto_stop_min,
                ephemeral=True,
            ),
            timeout=300,
        )
        try:
            # Apply repo-side perturbations to what the agent will read.
            for path, content in (file_overrides or {}).items():
                _upload_text(sb, f"/testbed/{path}", content)
            for path, content in (new_files or {}).items():
                _upload_text(sb, f"/testbed/{path}", content)

            def run_bash(cmd: str) -> str:
                try:
                    resp = sb.process.exec(cmd, cwd="/testbed", timeout=180)
                    out = resp.result or ""
                except Exception as e:  # keep the loop alive on a failed command
                    out = f"<command error: {e!r}>"
                return out[:4000]

            task = (f"Repository: {inst.repo} (checked out at /testbed).\n\n"
                    f"Bug report:\n{problem}\n\n"
                    "Explore /testbed, fix the bug in the source, then stop.")
            messages = [{"role": "user", "content": task}]

            for step in range(self.max_steps):
                resp = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=SYSTEM, tools=_TOOLS, messages=messages,
                )
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    break
                results = []
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "run_bash":
                        cmd = block.input.get("command", "")
                        self._log(f"step {step + 1}: {cmd[:80]}")
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": run_bash(cmd)})
                if results:
                    messages.append({"role": "user", "content": results})

            diff = sb.process.exec("git add -A && git diff --cached HEAD", cwd="/testbed",
                                   timeout=120)
            patch = diff.result or ""
            self._log(f"produced patch: {len(patch)} chars")
            return patch
        finally:
            try:
                sb.delete()
            except Exception:  # pragma: no cover
                pass
