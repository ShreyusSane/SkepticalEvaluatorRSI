"""The Skeptical Evaluator — perturbation-based reward modeling for a SWE-bench
code-fixing agent.

Package layout
--------------
- swebench_data : pull real SWE-bench instances (no Docker required)
- perturbations : the Perturbation Agent — finds concrete, meaning-preserving
                  places to alter the input given to a code-fixing agent
- fixers        : pluggable code-fixer interface (real adapters + simulated)
- evaluator     : the skeptical gate (mean + variance over perturbed re-checks)
- rsi_experiment: does perturbation-augmented training actually help a model learn?
"""

__all__ = [
    "swebench_data",
    "perturbations",
    "fixers",
    "evaluator",
    "rsi_experiment",
]
