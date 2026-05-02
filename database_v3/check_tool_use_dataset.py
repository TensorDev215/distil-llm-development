import argparse
import json
import re
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)


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


def extract_python_from_completion(completion: str):
    if not completion:
        return None

    match = TOOL_CALL_RE.search(completion)

    if not match:
        return None

    return match.group(1).strip()


def check_record(line_no: int, record: dict):
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    required_fields = [
        "prompt",
        "completion",
        "question",
        "gold",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    completion = record.get("completion", "")
    tool_call_field = record.get("tool_call")
    used_tool = record.get("used_tool", False)

    extracted_code = extract_python_from_completion(completion)

    if used_tool and not tool_call_field and not extracted_code:
        problems.append("used_tool=true but no tool_call and no <python> block found")

    if tool_call_field and not isinstance(tool_call_field, str):
        problems.append("tool_call exists but is not a string")

    if extracted_code and tool_call_field:
        if extracted_code.strip() != tool_call_field.strip():
            problems.append("tool_call field does not match <python>...</python> block")

    if used_tool and "<output>" not in completion:
        problems.append("used_tool=true but completion has no <output> block")

    if record.get("status") == "generated_with_tool":
        if not used_tool:
            problems.append("status=generated_with_tool but used_tool=false")
        if not tool_call_field and not extracted_code:
            problems.append("status=generated_with_tool but no Python code found")
        if record.get("tool_result") is None:
            problems.append("status=generated_with_tool but tool_result is null")

    return problems


def format_record_for_log(line_no: int, record: dict, problems: list[str]) -> str:
    completion = record.get("completion", "")
    extracted_code = extract_python_from_completion(completion)
    code = record.get("tool_call") or extracted_code

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"USED TOOL: {record.get('used_tool')}")
    lines.append("")

    lines.append("[QUESTION]")
    lines.append(str(record.get("question", "")).strip())
    lines.append("")

    lines.append("[GOLD ANSWER]")
    lines.append(str(record.get("gold", "")).strip())
    lines.append("")

    lines.append("[PYTHON CODE]")
    if code:
        lines.append(code.strip())
    else:
        lines.append("None")
    lines.append("")

    lines.append("[TOOL RESULT]")
    tool_result = record.get("tool_result")
    if tool_result is None:
        lines.append("None")
    else:
        lines.append(repr(tool_result))
    lines.append("")

    lines.append("[PROBLEMS]")
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("[COMPLETION TAIL]")
    if completion:
        lines.append(completion[-1000:].strip())
    else:
        lines.append("None")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check tool_use_dataset.jsonl records and log questions, answers, and Python tool calls."
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/tool_use_database_all_cases.jsonl",
        help="Path to tool_use_dataset.jsonl",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/tool_use_dataset_check.log",
        help="Path to output log file.",
    )

    parser.add_argument(
        "--show-python",
        action="store_true",
        help="Print extracted Python code block to terminal.",
    )

    parser.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Only print/log records with problems.",
    )

    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="Only print/log records that used a Python tool.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to inspect. 0 means all.",
    )

    args = parser.parse_args()

    path = Path(args.path)
    log_path = Path(args.log)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    logged_count = 0
    ok_count = 0
    bad_count = 0
    used_tool_count = 0
    python_block_count = 0
    status_counts = {}

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total += 1

            if args.limit and total > args.limit:
                break

            status = record.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            completion = record.get("completion", "")
            extracted_code = extract_python_from_completion(completion)

            used_tool = bool(record.get("used_tool"))

            if used_tool:
                used_tool_count += 1

            if extracted_code:
                python_block_count += 1

            problems = check_record(line_no, record)

            if problems:
                bad_count += 1
            else:
                ok_count += 1

            if args.show_errors_only and not problems:
                continue

            if args.tool_only and not used_tool:
                continue

            log_text = format_record_for_log(line_no, record, problems)
            log_f.write(log_text)
            log_f.write("\n")
            log_f.flush()

            logged_count += 1

            print("=" * 80)
            print(f"Line: {line_no}")
            print(f"Status: {status}")
            print(f"Used tool: {used_tool}")
            print(f"Question: {str(record.get('question', ''))[:180]}")
            print(f"Gold: {record.get('gold')}")

            if problems:
                print("Problems:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print("Problems: none")

            if args.show_python:
                code = record.get("tool_call") or extracted_code

                if code:
                    print("\nPython code:")
                    print("-" * 40)
                    print(code)
                    print("-" * 40)
                    print(f"Tool result: {repr(record.get('tool_result'))}")
                else:
                    print("\nPython code: none")

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        summary_lines.append(f"Total records checked: {total}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append(f"Records with used_tool=true: {used_tool_count}")
        summary_lines.append(f"Records with <python> block: {python_block_count}")
        summary_lines.append("")
        summary_lines.append("Status counts:")

        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status}: {count}")

        summary_text = "\n".join(summary_lines)

        log_f.write(summary_text)
        log_f.write("\n")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")


if __name__ == "__main__":
    main()