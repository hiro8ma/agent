"""The task an actor must solve, plus an objective evaluator.

The evaluator runs generated code against hidden test cases, so success / failure is
decided by execution — not by an LLM judge. This keeps the loop's reward signal
deterministic and lets the fake actor demonstrate genuine improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A small spec the actor sees. The function name and signature are fixed so the
# evaluator can locate and call the produced code.
TASK_PROMPT = """Write a Python function with this exact signature:

    def solve(nums: list[int]) -> int

It must return the sum of the EVEN numbers in `nums`. Return 0 when there are none.
Output only the function definition. No prose, no markdown fences.
"""

# (input, expected) pairs the evaluator checks. Hidden from the actor on purpose.
TEST_CASES: tuple[tuple[list[int], int], ...] = (
    ([1, 2, 3, 4], 6),
    ([], 0),
    ([1, 3, 5], 0),
    ([-2, -4, 1], -6),
    ([0, 0, 7], 0),
)


@dataclass
class EvalResult:
    """Outcome of running one attempt against the test suite."""

    passed: bool
    observation: str  # human-readable feedback fed into the reflector
    passed_count: int
    total: int


@dataclass
class CodingTask:
    """A self-graded coding task: a prompt the actor sees + a hidden test suite."""

    prompt: str = TASK_PROMPT
    function_name: str = "solve"
    test_cases: tuple[tuple[list[int], int], ...] = field(default=TEST_CASES)

    def evaluate(self, code: str) -> EvalResult:
        """Execute `code`, call the target function on every test case, summarise.

        The first failing or erroring case short-circuits into the observation so the
        reflector gets a concrete, actionable signal rather than a wall of output.
        """

        sandbox: dict[str, object] = {}
        try:
            compiled = compile(code, "<actor-attempt>", "exec")
            exec(compiled, sandbox)  # noqa: S102 - local demo sandbox, no untrusted input
        except SyntaxError as exc:
            return EvalResult(False, f"SyntaxError: {exc.msg} (line {exc.lineno})", 0, len(self.test_cases))
        except Exception as exc:  # noqa: BLE001 - report any load-time error verbatim
            return EvalResult(False, f"Error while defining function: {exc!r}", 0, len(self.test_cases))

        fn = sandbox.get(self.function_name)
        if not callable(fn):
            return EvalResult(
                False,
                f"No callable named `{self.function_name}` was defined.",
                0,
                len(self.test_cases),
            )

        passed = 0
        for nums, expected in self.test_cases:
            try:
                got = fn(list(nums))
            except Exception as exc:  # noqa: BLE001 - runtime crash is a failure signal
                return EvalResult(
                    False,
                    f"Crashed on input {nums!r}: {exc!r}. Passed {passed}/{len(self.test_cases)}.",
                    passed,
                    len(self.test_cases),
                )
            if got != expected:
                return EvalResult(
                    False,
                    (
                        f"Wrong output on input {nums!r}: expected {expected!r}, got {got!r}. "
                        f"Passed {passed}/{len(self.test_cases)} before this."
                    ),
                    passed,
                    len(self.test_cases),
                )
            passed += 1

        return EvalResult(True, f"All {passed}/{len(self.test_cases)} cases passed.", passed, len(self.test_cases))


def default_task() -> CodingTask:
    return CodingTask()
