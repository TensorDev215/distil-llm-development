#!/usr/bin/env python3

import argparse
import json
import random
import sys
import traceback
from pathlib import Path


CORRECTION_STREAM_SEED = 0xC0FFEE29


BUG_KINDS = [
    "off_by_one_range",
    "swap_subtract",
    "wrong_comparator",
    "wrong_init",
    "early_break",
    "wrong_index",
    "wrong_modulo",
    "missing_edge_case",
]


CONSONANTS = [
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
    "n", "p", "r", "s", "t", "v", "w", "z",
    "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
    "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr",
    "th", "sh", "ch",
]

VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "ou", "ay"]

CODAS = ["", "n", "r", "l", "s", "t", "ck", "rd", "rt", "ng", "st", "ld"]


def synthetic_syllable(r: random.Random) -> str:
    return r.choice(CONSONANTS) + r.choice(VOWELS) + r.choice(CODAS)


def synthetic_word(r: random.Random, n_syllables: int = 1) -> str:
    return "".join(synthetic_syllable(r) for _ in range(n_syllables))


def comment_block(text: str) -> str:
    return "\n".join("# " + line if line else "#" for line in text.splitlines())


def comment_tests(tests: list[str]) -> str:
    return "\n".join(
        ("# " + line.lstrip()) if line.startswith("    ") else ("# " + line)
        for line in tests
    )


def normalize_single_line_completion(completion: str) -> str:
    """
    Keep completion compatible with the existing pod_eval_vllm.py /
    humaneval_sandbox checking path.

    Important:
    - Completion must be function body only.
    - Completion must be one physical non-empty line.
    - Completion must be indented because the prompt already contains
      the function signature and docstring.
    """
    completion = completion.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in completion.splitlines() if line.strip()]

    if len(lines) != 1:
        raise ValueError(
            "Completion must be exactly one non-empty line for this database. "
            f"Got {len(lines)} non-empty lines: {completion!r}"
        )

    line = lines[0].strip()

    if line.startswith("def "):
        raise ValueError("Completion must be function body only, not a full def.")

    return "    " + line + "\n"


def make_prompt(
    buggy_code: str,
    error_trace: str,
    tests: list[str],
    signature_with_docstring: str,
) -> str:
    buggy_commented = comment_block(buggy_code)
    error_commented = comment_block(error_trace)
    commented_tests = comment_tests(tests)

    return (
        "# The following Python function fails its tests with the\n"
        "# error trace shown below. The intended behaviour is in the\n"
        "# corrected docstring. Read the trace, identify the line\n"
        "# that produces the wrong value, and write the CORRECTED\n"
        "# implementation. Output only the function body, no extra\n"
        "# explanation, no markdown fences.\n"
        "#\n"
        "# Buggy version, DO NOT include in your output:\n"
        f"{buggy_commented}\n"
        "#\n"
        "# Test failure trace:\n"
        f"{error_commented}\n"
        "#\n"
        "# All tests the corrected version must pass:\n"
        f"{commented_tests}\n"
        "#\n"
        "# Now complete the corrected version below.\n"
        f"{signature_with_docstring}"
    )


def make_test_block(tests: list[str]) -> str:
    return "def check(candidate):\n" + "\n".join(tests) + "\n"


def make_record(
    kind: str,
    index: int,
    entry_point: str,
    prompt: str,
    completion: str,
    test: str,
    buggy_code: str,
    error_trace: str,
    seed: int,
) -> dict:
    return {
        "src": f"procedural_correction/{kind}",
        "kind": kind,
        "task_id": f"correction/{kind}/{index:05d}",
        "prompt": prompt,
        "completion": normalize_single_line_completion(completion),
        "test": test,
        "entry_point": entry_point,
        "buggy_code": buggy_code,
        "error_trace": error_trace,
        "status": "gold_correction",
        "seed": seed,
    }


def rename_entry(
    original_entry: str,
    buggy: str,
    sig: str,
    error_trace: str,
    r: random.Random,
) -> tuple[str, str, str, str]:
    suffix = "_" + synthetic_word(r, 1)
    entry = original_entry + suffix

    buggy = buggy.replace(f"def {original_entry}(", f"def {entry}(")
    sig = sig.replace(f"def {original_entry}(", f"def {entry}(")

    # Rename function references inside traces such as:
    #   in first_or_default
    error_trace = error_trace.replace(f"in {original_entry}\n", f"in {entry}\n")

    return entry, buggy, sig, error_trace


def generate_item(kind: str, index: int, seed: int) -> dict:
    r = random.Random(seed)

    if kind == "off_by_one_range":
        original_entry = "sum_to"
        n_t = r.randint(5, 15)

        buggy = (
            "def sum_to(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        )

        sig = (
            "def sum_to(n: int) -> int:\n"
            '    """Return the sum of integers from 1 to n inclusive.\n'
            "    Examples: sum_to(3) == 6 (1+2+3); sum_to(5) == 15.\n"
            '    """\n'
        )

        expected = sum(range(1, n_t + 1))
        expected_next = sum(range(1, n_t + 2))
        actual_buggy = sum(range(1, n_t))

        tests = [
            f"    assert candidate({n_t}) == {expected}",
            f"    assert candidate({n_t + 1}) == {expected_next}",
            "    assert candidate(1) == 1",
            "    assert candidate(2) == 3",
        ]

        error_trace = (
            '  File "test_sum_to.py", line 3, in test_sum_to\n'
            f"    assert candidate({n_t}) == {expected}\n"
            f"AssertionError: expected {expected}, "
            f"got {actual_buggy} (off by {expected - actual_buggy})"
        )

        # Single-line corrected body for sandbox compatibility.
        completion = "return sum(range(1, n + 1))"

    elif kind == "swap_subtract":
        original_entry = "first_minus_second"

        buggy = (
            "def first_minus_second(a: int, b: int) -> int:\n"
            "    return b - a\n"
        )

        sig = (
            "def first_minus_second(a: int, b: int) -> int:\n"
            '    """Return a minus b, i.e. a - b."""\n'
        )

        tests = [
            "    assert candidate(10, 3) == 7",
            "    assert candidate(0, 5) == -5",
            "    assert candidate(-4, -2) == -2",
            "    assert candidate(100, 99) == 1",
        ]

        error_trace = (
            '  File "test_subtract.py", line 3, in test_subtract\n'
            "    assert candidate(10, 3) == 7\n"
            "AssertionError: expected 7, got -7 (sign reversed)"
        )

        completion = "return a - b"

    elif kind == "wrong_comparator":
        original_entry = "at_least"
        threshold = r.randint(3, 9)

        buggy = (
            f"def at_least(arr, threshold={threshold}):\n"
            "    return sum(1 for x in arr if x > threshold)\n"
        )

        sig = (
            f"def at_least(arr, threshold={threshold}):\n"
            '    """Count elements in arr that are >= threshold, inclusive."""\n'
        )

        tests = [
            f"    assert candidate([1, {threshold}, {threshold - 1}, {threshold + 1}, {threshold}], threshold={threshold}) == 3",
            f"    assert candidate([{threshold}, {threshold}, {threshold}], threshold={threshold}) == 3",
            f"    assert candidate([0, 1, 2], threshold={threshold}) == 0",
            "    assert candidate([], threshold=1) == 0",
        ]

        error_trace = (
            '  File "test_at_least.py", line 3, in test_at_least\n'
            f"    assert candidate([1, {threshold}, {threshold - 1}, {threshold + 1}, {threshold}], threshold={threshold}) == 3\n"
            "AssertionError: expected 3, got 1 (only counted strictly-greater values)"
        )

        completion = "return sum(1 for x in arr if x >= threshold)"

    elif kind == "wrong_init":
        original_entry = "product_of_list"

        buggy = (
            "def product_of_list(arr):\n"
            "    total = 0\n"
            "    for x in arr:\n"
            "        total *= x\n"
            "    return total\n"
        )

        sig = (
            "def product_of_list(arr: list[int]) -> int:\n"
            '    """Return the product of integers in arr. Empty list returns 1."""\n'
        )

        tests = [
            "    assert candidate([2, 3, 4]) == 24",
            "    assert candidate([5]) == 5",
            "    assert candidate([]) == 1",
            "    assert candidate([1, 2, 3, 4, 5]) == 120",
        ]

        error_trace = (
            '  File "test_product.py", line 1, in test_product\n'
            "    assert candidate([2, 3, 4]) == 24\n"
            "AssertionError: expected 24, got 0 (accumulator initialised wrong)"
        )

        # Single physical line, but two valid Python statements.
        # math.prod([]) returns 1, matching the docstring.
        completion = "import math; return math.prod(arr)"

    elif kind == "early_break":
        original_entry = "find_largest"

        buggy = (
            "def find_largest(arr):\n"
            "    largest = arr[0]\n"
            "    for x in arr:\n"
            "        if x > largest:\n"
            "            largest = x\n"
            "            break\n"
            "    return largest\n"
        )

        sig = (
            "def find_largest(arr: list[int]) -> int:\n"
            '    """Return the largest integer in arr. Assume non-empty."""\n'
        )

        tests = [
            "    assert candidate([3, 7, 2, 9, 4]) == 9",
            "    assert candidate([1, 2, 3, 4, 5, 6]) == 6",
            "    assert candidate([5, 4, 3, 2, 1]) == 5",
            "    assert candidate([42]) == 42",
        ]

        error_trace = (
            '  File "test_largest.py", line 1, in test_largest\n'
            "    assert candidate([3, 7, 2, 9, 4]) == 9\n"
            "AssertionError: expected 9, got 7 (loop exited too early)"
        )

        completion = "return max(arr)"

    elif kind == "wrong_index":
        original_entry = "second_last"
        sample_arr = [r.randint(1, 99) for _ in range(r.randint(5, 9))]

        buggy = (
            "def second_last(arr):\n"
            "    return arr[-1]\n"
        )

        sig = (
            "def second_last(arr: list):\n"
            '    """Return the second-to-last element of arr."""\n'
        )

        tests = [
            f"    assert candidate({sample_arr!r}) == {sample_arr[-2]}",
            "    assert candidate([1, 2]) == 1",
            "    assert candidate(['a', 'b', 'c']) == 'b'",
            "    assert candidate([10, 20, 30, 40, 50]) == 40",
        ]

        error_trace = (
            '  File "test_second_last.py", line 1, in test_second_last\n'
            f"    assert candidate({sample_arr!r}) == {sample_arr[-2]}\n"
            f"AssertionError: expected {sample_arr[-2]}, got {sample_arr[-1]} (off-by-one index)"
        )

        completion = "return arr[-2]"

    elif kind == "wrong_modulo":
        original_entry = "mod_then_double"
        mod = r.choice([7, 11, 13])
        wrong_mod = mod + 1

        buggy = (
            f"def mod_then_double(n, modulus={mod}):\n"
            f"    return 2 * (n % {wrong_mod})\n"
        )

        sig = (
            f"def mod_then_double(n: int, modulus: int = {mod}) -> int:\n"
            '    """Return 2 * (n % modulus)."""\n'
        )

        tests = [
            f"    assert candidate(10, modulus={mod}) == {2 * (10 % mod)}",
            f"    assert candidate({mod * 2 + 3}, modulus={mod}) == {2 * ((mod * 2 + 3) % mod)}",
            f"    assert candidate(0, modulus={mod}) == 0",
            f"    assert candidate({mod - 1}, modulus={mod}) == {2 * (mod - 1)}",
        ]

        error_trace = (
            '  File "test_mod.py", line 1, in test_mod\n'
            f"    assert candidate(10, modulus={mod}) == {2 * (10 % mod)}\n"
            f"AssertionError: expected {2 * (10 % mod)}, got {2 * (10 % wrong_mod)} "
            f"(used wrong divisor: {wrong_mod} instead of {mod})"
        )

        completion = "return 2 * (n % modulus)"

    elif kind == "missing_edge_case":
        original_entry = "first_or_default"

        buggy = (
            "def first_or_default(arr, default=None):\n"
            "    return arr[0]\n"
        )

        sig = (
            "def first_or_default(arr: list, default=None):\n"
            '    """Return arr[0] if non-empty, else return default."""\n'
        )

        tests = [
            "    assert candidate([7, 8, 9]) == 7",
            "    assert candidate([]) is None",
            "    assert candidate([], default=-1) == -1",
            "    assert candidate(['a']) == 'a'",
        ]

        error_trace = (
            '  File "test_default.py", line 2, in test_default\n'
            "    assert candidate([]) is None\n"
            '  File "<solution>", line 2, in first_or_default\n'
            "    return arr[0]\n"
            "IndexError: list index out of range"
        )

        completion = "return arr[0] if arr else default"

    else:
        raise ValueError(f"Unknown correction kind: {kind}")

    entry_point, buggy, sig, error_trace = rename_entry(
        original_entry=original_entry,
        buggy=buggy,
        sig=sig,
        error_trace=error_trace,
        r=r,
    )

    prompt = make_prompt(
        buggy_code=buggy,
        error_trace=error_trace,
        tests=tests,
        signature_with_docstring=sig,
    )

    test_block = make_test_block(tests)

    return make_record(
        kind=kind,
        index=index,
        entry_point=entry_point,
        prompt=prompt,
        completion=completion,
        test=test_block,
        buggy_code=buggy,
        error_trace=error_trace,
        seed=seed,
    )


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random((int(seed) ^ CORRECTION_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in BUG_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)
            record = generate_item(kind, index=index, seed=item_seed)
            records.append(record)
            index += 1

    if shuffle:
        main_rng.shuffle(records)

    return records


def write_jsonl(records: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_record(record: dict) -> tuple[bool, str]:
    """
    Local quick verification.

    This does not replace your pod_eval_vllm.py / humaneval_sandbox check.
    It just catches obvious generation bugs before writing or after generation.
    """
    namespace = {}

    solution_code = record["prompt"] + record["completion"]
    test_code = record["test"]

    try:
        exec(solution_code, namespace)
        exec(test_code, namespace)

        entry_point = record["entry_point"]
        candidate = namespace[entry_point]
        namespace["check"](candidate)

        return True, ""

    except Exception:
        return False, traceback.format_exc()


def verify_records(records: list[dict], max_failures: int = 10) -> bool:
    failures = []

    for record in records:
        ok, err = verify_record(record)

        if not ok:
            failures.append((record["task_id"], record["kind"], err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated corrections passed tests.")
        return True

    print("Local verification failed.")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, err in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(err)

    return False


def assert_all_completions_one_line(records: list[dict]) -> None:
    bad = []

    for record in records:
        completion = record["completion"]
        nonempty_lines = [line for line in completion.splitlines() if line.strip()]

        if len(nonempty_lines) != 1:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    len(nonempty_lines),
                    completion,
                )
            )

    if bad:
        print("Found non-one-line completions:", file=sys.stderr)

        for task_id, kind, n_lines, completion in bad[:10]:
            print("=" * 80, file=sys.stderr)
            print(f"Task: {task_id}", file=sys.stderr)
            print(f"Kind: {kind}", file=sys.stderr)
            print(f"Non-empty lines: {n_lines}", file=sys.stderr)
            print(repr(completion), file=sys.stderr)

        raise SystemExit(1)


def print_counts(records: list[dict]) -> None:
    counts = {}

    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1

    print("\nKind counts:")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic correction JSONL database with one-line "
            "gold fixed function bodies compatible with pod_eval_vllm.py / "
            "humaneval_sandbox checking."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/correction_database_all_cases1.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260501,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-per-kind",
        type=int,
        default=10,
        help="Number of correction records to generate per bug kind.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle final records.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting it.",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run local verification that prompt + completion passes tests.",
    )

    args = parser.parse_args()

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        shuffle=not args.no_shuffle,
    )

    assert_all_completions_one_line(records)

    if args.verify:
        ok = verify_records(records)

        if not ok:
            print("Not writing JSONL because local verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"Bug kinds: {len(BUG_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print("Completion format: one-line function body only")

    print_counts(records)


if __name__ == "__main__":
    main()
