#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the repository root containing ./distil.
    This lets the script run from project root or database_v3/.
    """
    current = start.resolve()

    for path in [current] + list(current.parents):
        if (path / "distil").exists():
            return path

    raise RuntimeError(
        "Could not find repository root containing ./distil. "
        "Run this script from inside your distil-llm-development repo."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))


import distil.scripts.humaneval_sandbox as hs
import distil.scripts.pod_eval_vllm as eval_node


REQUIRED_FIELDS = [
    "prompt",
    "completion",
    "test",
    "entry_point",
]


DEFAULT_INPUT = "dataset/debug_database_all_cases.jsonl"
DEFAULT_FAILURE_LOG = "database_v3/debug_database_failures.log"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")

            if not raw.strip():
                continue

            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": str(e),
                    "_raw": raw[:1000],
                }


def strip_markdown_fences(text: str) -> str:
    """
    Defensive cleanup only. Your generated database should not contain fences.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.strip()

    if not stripped.startswith("```"):
        return text

    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip("\n") + "\n"


def clean_completion(completion: str) -> str:
    """
    Match the same cleanup style used around pod_eval_vllm debug answering.

    Important:
    This checker does NOT assemble or rewrite the solution.
    It passes:

        prompt, cleaned_completion, test, entry_point

    directly to humaneval_sandbox.run_batch.
    """
    completion = str(completion)
    completion = eval_node._strip_thinking_probe(completion)
    completion = strip_markdown_fences(completion)

    return completion


def get_kind(record: dict) -> str:
    if record.get("kind"):
        return str(record["kind"])

    src = str(record.get("src", ""))

    if "/" in src:
        return src.rsplit("/", 1)[-1]

    return "unknown"


def validate_record(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing field: {field}")

    if problems:
        return problems

    prompt = record.get("prompt")
    completion = record.get("completion")
    test = record.get("test")
    entry_point = record.get("entry_point")

    if not isinstance(prompt, str):
        problems.append("prompt is not a string")

    if not isinstance(completion, str):
        problems.append("completion is not a string")

    if not isinstance(test, str):
        problems.append("test is not a string")

    if not isinstance(entry_point, str):
        problems.append("entry_point is not a string")

    if isinstance(prompt, str):
        if "def " not in prompt:
            problems.append("prompt does not appear to contain a function signature")

        if "# Now complete the corrected version below." not in prompt:
            problems.append("prompt does not contain expected debug completion marker")

    if isinstance(test, str):
        if "def check(candidate):" not in test:
            problems.append("test does not contain def check(candidate):")

    if isinstance(entry_point, str):
        if not entry_point.strip():
            problems.append("entry_point is empty")

    if isinstance(completion, str):
        cleaned = clean_completion(completion)
        nonempty_lines = [line for line in cleaned.splitlines() if line.strip()]

        if not cleaned.strip():
            problems.append("completion is empty after cleanup")

        if completion.strip().startswith("```"):
            problems.append("completion contains markdown fence")

        if completion.lstrip().startswith("def "):
            problems.append("completion includes full function definition, expected body only")

        # For this database we expect one-line gold bodies because that is
        # the most stable format for your current sandbox/checking path.
        if len(nonempty_lines) != 1:
            problems.append(
                f"completion should contain exactly one non-empty line, got {len(nonempty_lines)}"
            )

    return problems


def make_sandbox_item(record: dict) -> tuple[str, str, str, str]:
    """
    This is the key compatibility path.

    Same shape as your debug-builder sample:

        hs.run_batch([
            (
                item["prompt"],
                eval_node._strip_thinking_probe(answer),
                item["test"],
                item["entry_point"],
            )
        ])
    """
    prompt = record["prompt"]
    completion = clean_completion(record["completion"])
    test = record["test"]
    entry_point = record["entry_point"]

    return prompt, completion, test, entry_point


def run_sandbox_batch(records: list[dict], max_workers: int):
    items = [make_sandbox_item(record) for record in records]

    return hs.run_batch(
        items,
        max_workers=max_workers,
    )


def result_passed(result) -> bool:
    return bool(result and getattr(result, "passed", False))


def result_to_text(result) -> str:
    if result is None:
        return "No result returned by sandbox."

    fields = []

    for attr in [
        "passed",
        "result",
        "error",
        "traceback",
        "stdout",
        "stderr",
        "timeout",
    ]:
        if hasattr(result, attr):
            value = getattr(result, attr)

            if value:
                fields.append(f"{attr}: {value}")

    if not fields:
        return repr(result)

    return "\n".join(fields)


def write_failure_log(log_path: Path, failures: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        for failure in failures:
            record = failure["record"]

            f.write("=" * 100 + "\n")
            f.write(f"Line: {failure['line_no']}\n")
            f.write(f"Task ID: {record.get('task_id')}\n")
            f.write(f"Kind: {get_kind(record)}\n")
            f.write(f"Entry point: {record.get('entry_point')}\n")
            f.write(f"Src: {record.get('src')}\n")
            f.write(f"Status: {record.get('status')}\n")
            f.write(f"Block seed: {record.get('block_seed')}\n")
            f.write(f"Index: {record.get('index')}\n")
            f.write("\n")

            if failure.get("validation_problems"):
                f.write("[VALIDATION PROBLEMS]\n")

                for problem in failure["validation_problems"]:
                    f.write(f"- {problem}\n")

                f.write("\n")

            if failure.get("sandbox_result"):
                f.write("[SANDBOX RESULT]\n")
                f.write(failure["sandbox_result"])
                f.write("\n\n")

            f.write("[PROMPT]\n")
            f.write(str(record.get("prompt", "")))
            f.write("\n\n")

            f.write("[RAW COMPLETION]\n")
            f.write(str(record.get("completion", "")))
            f.write("\n\n")

            f.write("[CLEANED COMPLETION]\n")
            f.write(clean_completion(record.get("completion", "")))
            f.write("\n\n")

            f.write("[TEST]\n")
            f.write(str(record.get("test", "")))
            f.write("\n\n")

            f.write("[ENTRY POINT]\n")
            f.write(str(record.get("entry_point", "")))
            f.write("\n\n")

            if record.get("buggy_code"):
                f.write("[BUGGY CODE]\n")
                f.write(str(record.get("buggy_code", "")))
                f.write("\n\n")


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}:")

    if not counter:
        print("  <none>")
        return

    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check debug JSONL database using pod_eval_vllm.py / "
            "humaneval_sandbox path."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to debug JSONL database.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of sandbox workers.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of records to evaluate per sandbox batch.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to check. 0 means all.",
    )

    parser.add_argument(
        "--failure-log",
        type=str,
        default=DEFAULT_FAILURE_LOG,
        help="Path to write failure diagnostics.",
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failure details to terminal.",
    )

    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help=(
            "Treat all validation problems as fatal. Without this flag, "
            "style problems are logged but sandbox still runs where possible."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.path)

    if not input_path.exists():
        raise FileNotFoundError(f"Debug database file does not exist: {input_path}")

    total = 0
    validation_bad = 0
    sandbox_passed = 0
    sandbox_failed = 0

    status_counts = Counter()
    kind_counts = Counter()
    kind_pass_counts = Counter()
    kind_fail_counts = Counter()
    validation_problem_counts = Counter()

    failures = []

    pending_records = []
    pending_line_nos = []

    def add_failure(
        line_no: int,
        record: dict,
        validation_problems: list[str] | None = None,
        sandbox_result: str | None = None,
    ) -> None:
        failure = {
            "line_no": line_no,
            "record": record,
        }

        if validation_problems:
            failure["validation_problems"] = validation_problems

            for problem in validation_problems:
                validation_problem_counts[problem] += 1

        if sandbox_result:
            failure["sandbox_result"] = sandbox_result

        failures.append(failure)

    def is_fatal_validation_problem(problems: list[str]) -> bool:
        hard_prefixes = [
            "JSON decode error",
            "missing field",
            "prompt is not a string",
            "completion is not a string",
            "test is not a string",
            "entry_point is not a string",
            "completion is empty",
        ]

        return any(
            any(problem.startswith(prefix) for prefix in hard_prefixes)
            for problem in problems
        )

    def flush_batch() -> None:
        nonlocal sandbox_passed
        nonlocal sandbox_failed

        if not pending_records:
            return

        results = run_sandbox_batch(
            pending_records,
            max_workers=args.max_workers,
        )

        for line_no, record, result in zip(pending_line_nos, pending_records, results):
            kind = get_kind(record)

            if result_passed(result):
                sandbox_passed += 1
                kind_pass_counts[kind] += 1
            else:
                sandbox_failed += 1
                kind_fail_counts[kind] += 1

                sandbox_text = result_to_text(result)

                add_failure(
                    line_no=line_no,
                    record=record,
                    sandbox_result=sandbox_text,
                )

                if args.show_failures:
                    print("=" * 80)
                    print(
                        f"FAILED line={line_no} "
                        f"task_id={record.get('task_id')} "
                        f"kind={kind}"
                    )
                    print(sandbox_text)

        pending_records.clear()
        pending_line_nos.clear()

    for line_no, record in load_jsonl(input_path):
        total += 1

        if args.limit and total > args.limit:
            total -= 1
            break

        if "_json_error" not in record:
            kind = get_kind(record)
            status = str(record.get("status", "unknown"))

            kind_counts[kind] += 1
            status_counts[status] += 1

        problems = validate_record(record)

        if problems:
            fatal = args.strict_validation or is_fatal_validation_problem(problems)

            if fatal:
                validation_bad += 1

                add_failure(
                    line_no=line_no,
                    record=record,
                    validation_problems=problems,
                )

                if args.show_failures:
                    print("=" * 80)
                    print(f"VALIDATION FAILED line={line_no}")

                    for problem in problems:
                        print(f"- {problem}")

                continue

            # Non-fatal validation warnings are counted, but sandbox still runs.
            for problem in problems:
                validation_problem_counts[problem] += 1

        pending_records.append(record)
        pending_line_nos.append(line_no)

        if len(pending_records) >= args.batch_size:
            flush_batch()

    flush_batch()

    failure_log = Path(args.failure_log)

    if failures:
        write_failure_log(failure_log, failures)

    print("=" * 80)
    print("DEBUG DATABASE CHECK SUMMARY")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Total records checked: {total}")
    print(f"Validation bad records: {validation_bad}")
    print(f"Sandbox passed: {sandbox_passed}")
    print(f"Sandbox failed: {sandbox_failed}")
    print(f"Total failures logged: {len(failures)}")

    if failures:
        print(f"Failure log: {failure_log}")

    print_counter("Status counts", status_counts)

    print("\nKind counts:")
    if not kind_counts:
        print("  <none>")
    else:
        for kind, count in sorted(kind_counts.items()):
            passed = kind_pass_counts.get(kind, 0)
            failed = kind_fail_counts.get(kind, 0)
            print(f"  {kind}: total={count}, passed={passed}, failed={failed}")

    if validation_problem_counts:
        print_counter("Validation problem counts", validation_problem_counts)

    if validation_bad == 0 and sandbox_failed == 0:
        print("\nResult: PASS")
    else:
        print("\nResult: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
