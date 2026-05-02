#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the repository root containing the distil package.
    This makes the script robust whether it is launched from project root,
    database_v3/, or somewhere else.
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


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": str(e),
                    "_raw": line[:500],
                }


def validate_record(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing field: {field}")

    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    test = record.get("test", "")
    entry_point = record.get("entry_point", "")

    if prompt and not isinstance(prompt, str):
        problems.append("prompt is not a string")

    if completion and not isinstance(completion, str):
        problems.append("completion is not a string")

    if test and not isinstance(test, str):
        problems.append("test is not a string")

    if entry_point and not isinstance(entry_point, str):
        problems.append("entry_point is not a string")

    if isinstance(prompt, str) and "def " not in prompt:
        problems.append("prompt does not appear to contain a function signature")

    if isinstance(test, str) and "def check(candidate):" not in test:
        problems.append("test does not contain def check(candidate):")

    if isinstance(completion, str):
        stripped = completion.strip()

        if not stripped:
            problems.append("completion is empty")

        if stripped.startswith("```"):
            problems.append("completion appears to contain markdown fence")

        if "def " in stripped.splitlines()[0]:
            problems.append("completion appears to include full function definition, expected body only")

    return problems


def make_eval_tuple(record: dict, strip_thinking: bool = True):
    prompt = record["prompt"]
    completion = record["completion"]
    test = record["test"]
    entry_point = record["entry_point"]

    if strip_thinking:
        completion = eval_node._strip_thinking_probe(completion)

    return prompt, completion, test, entry_point


def run_sandbox_batch(records: list[dict], max_workers: int):
    items = [
        make_eval_tuple(record)
        for record in records
    ]

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
            f.write(f"Kind: {record.get('kind')}\n")
            f.write(f"Entry point: {record.get('entry_point')}\n")
            f.write(f"Seed: {record.get('seed')}\n")
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

            f.write("[COMPLETION]\n")
            f.write(str(record.get("completion", "")))
            f.write("\n\n")

            f.write("[TEST]\n")
            f.write(str(record.get("test", "")))
            f.write("\n\n")

            if record.get("buggy_code"):
                f.write("[BUGGY CODE]\n")
                f.write(str(record.get("buggy_code", "")))
                f.write("\n\n")

            if record.get("error_trace"):
                f.write("[ERROR TRACE]\n")
                f.write(str(record.get("error_trace", "")))
                f.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check correction JSONL database using pod_eval_vllm/humaneval sandbox path."
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/correction_database_all_cases1.jsonl",
        help="Path to correction JSONL database.",
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
        default="database_v3/correction_database_failures.log",
        help="Path to write failed records and diagnostics.",
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failed records to terminal.",
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    total = 0
    validation_bad = 0
    sandbox_passed = 0
    sandbox_failed = 0
    status_counts = {}
    kind_counts = {}
    kind_pass_counts = {}
    kind_fail_counts = {}

    failures = []

    pending_records = []
    pending_line_nos = []

    def flush_batch():
        nonlocal sandbox_passed
        nonlocal sandbox_failed

        if not pending_records:
            return

        results = run_sandbox_batch(
            pending_records,
            max_workers=args.max_workers,
        )

        for line_no, record, result in zip(pending_line_nos, pending_records, results):
            kind = record.get("kind", "unknown")

            if result_passed(result):
                sandbox_passed += 1
                kind_pass_counts[kind] = kind_pass_counts.get(kind, 0) + 1
            else:
                sandbox_failed += 1
                kind_fail_counts[kind] = kind_fail_counts.get(kind, 0) + 1

                failure = {
                    "line_no": line_no,
                    "record": record,
                    "sandbox_result": result_to_text(result),
                }
                failures.append(failure)

                if args.show_failures:
                    print("=" * 80)
                    print(f"FAILED line={line_no} task_id={record.get('task_id')} kind={kind}")
                    print(result_to_text(result))

        pending_records.clear()
        pending_line_nos.clear()

    for line_no, record in load_jsonl(path):
        total += 1

        if args.limit and total > args.limit:
            break

        if "_json_error" not in record:
            status = record.get("status", "unknown")
            kind = record.get("kind", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

        problems = validate_record(record)

        if problems:
            validation_bad += 1
            failures.append(
                {
                    "line_no": line_no,
                    "record": record,
                    "validation_problems": problems,
                }
            )

            if args.show_failures:
                print("=" * 80)
                print(f"VALIDATION FAILED line={line_no}")
                for problem in problems:
                    print(f"- {problem}")

            continue

        pending_records.append(record)
        pending_line_nos.append(line_no)

        if len(pending_records) >= args.batch_size:
            flush_batch()

    flush_batch()

    failure_log = Path(args.failure_log)

    if failures:
        write_failure_log(failure_log, failures)

    print("=" * 80)
    print("CORRECTION DATABASE CHECK SUMMARY")
    print("=" * 80)
    print(f"Input file: {path}")
    print(f"Total records checked: {total}")
    print(f"Validation bad records: {validation_bad}")
    print(f"Sandbox passed: {sandbox_passed}")
    print(f"Sandbox failed: {sandbox_failed}")
    print(f"Total failures logged: {len(failures)}")

    if failures:
        print(f"Failure log: {failure_log}")

    print("\nStatus counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nKind counts:")
    for kind, count in sorted(kind_counts.items()):
        passed = kind_pass_counts.get(kind, 0)
        failed = kind_fail_counts.get(kind, 0)
        print(f"  {kind}: total={count}, passed={passed}, failed={failed}")

    if validation_bad == 0 and sandbox_failed == 0:
        print("\nResult: PASS")
    else:
        print("\nResult: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
