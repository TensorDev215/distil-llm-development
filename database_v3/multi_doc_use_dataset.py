#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path


AXE_STREAM_SEED = 0xA0CE2026


AXE_KINDS = [
    "single_doc_lookup",
    "cross_doc_join",
    "latest_fact_wins",
    "count_matching_docs",
    "numeric_max",
    "numeric_min",
    "set_intersection",
    "distractor_filter",
    "multi_hop_owner_item",
    "date_order_lookup",
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


COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "silver", "black", "white", "teal",
]

CITIES = [
    "Aster Bay", "Northwick", "Ravenford", "Clearhaven", "Stonebridge",
    "Lakepoint", "Elmreach", "Brightfall", "Westmere", "Ironvale",
    "Maple Junction", "Harbor Glen", "Cedar Hollow", "Silver Ridge",
]

DEPARTMENTS = [
    "logistics", "finance", "research", "support", "marketing",
    "security", "operations", "analytics", "design", "training",
]

ITEMS = [
    "laptop", "tablet", "scanner", "router", "printer",
    "monitor", "camera", "keyboard", "badge", "projector",
    "headset", "dock", "charger", "microphone",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def synthetic_syllable(r: random.Random) -> str:
    return r.choice(CONSONANTS) + r.choice(VOWELS) + r.choice(CODAS)


def synthetic_word(r: random.Random, n_syllables: int = 1) -> str:
    return "".join(synthetic_syllable(r) for _ in range(n_syllables))


def synthetic_name(r: random.Random) -> str:
    first = synthetic_word(r, r.choice([1, 2])).capitalize()
    last = synthetic_word(r, r.choice([1, 2])).capitalize()
    return f"{first} {last}"


def normalize_single_line_answer(answer: str) -> str:
    """
    Keep completion compatible with simple supervised fine-tuning.

    Rules:
    - Completion is answer only.
    - Completion is exactly one physical non-empty line.
    - Completion ends with a newline.
    - No markdown fences.
    - No extra explanation.
    """
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in answer.splitlines() if line.strip()]

    if len(lines) != 1:
        raise ValueError(
            "Answer must be exactly one non-empty line for this database. "
            f"Got {len(lines)} non-empty lines: {answer!r}"
        )

    line = lines[0].strip()

    if line.startswith("```"):
        raise ValueError("Answer must not include markdown fences.")

    return line + "\n"


def normalize_for_match(text: str) -> str:
    """
    Normalize text for local exact-match verification.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    return text


def make_doc(doc_id: int, title: str, text: str) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "text": text,
    }


def make_document_block(documents: list[dict]) -> str:
    blocks = []

    for doc in documents:
        blocks.append(
            f"Document {doc['id']}:\n"
            f"Title: {doc['title']}\n"
            f"{doc['text']}"
        )

    return "\n\n".join(blocks)


def make_prompt(
    documents: list[dict],
    question: str,
) -> str:
    document_block = make_document_block(documents)

    return (
        "You are given several short documents.\n"
        "Answer the question using only the information in the documents.\n"
        "Output only the final answer.\n"
        "Do not include extra explanation or markdown fences.\n"
        "\n"
        f"{document_block}\n"
        "\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_record(
    kind: str,
    index: int,
    prompt: str,
    completion: str,
    answer: str,
    documents: list[dict],
    question: str,
    seed: int,
) -> dict:
    return {
        "src": f"procedural_multi_doc_axe/{kind}",
        "kind": kind,
        "task_id": f"multi_doc_axe/{kind}/{index:05d}",
        "prompt": prompt,
        "completion": normalize_single_line_answer(completion),
        "answer": answer,
        "documents": documents,
        "question": question,
        "status": "gold_axe_training",
        "seed": seed,
    }


def record_to_messages(record: dict) -> dict:
    return {
        "src": record["src"],
        "kind": record["kind"],
        "task_id": record["task_id"],
        "messages": [
            {
                "role": "user",
                "content": record["prompt"],
            },
            {
                "role": "assistant",
                "content": record["completion"].strip(),
            },
        ],
        "answer": record["answer"],
        "documents": record["documents"],
        "question": record["question"],
        "status": record["status"],
        "seed": record["seed"],
    }


def generate_item(kind: str, index: int, seed: int) -> dict:
    r = random.Random(seed)

    if kind == "single_doc_lookup":
        person = synthetic_name(r)
        city = r.choice(CITIES)
        color = r.choice(COLORS)
        distractor = synthetic_name(r)

        documents = [
            make_doc(
                1,
                "Staff Profile",
                f"{person} works in {city}. Their access badge is {color}.",
            ),
            make_doc(
                2,
                "Visitor Note",
                f"{distractor} visited {r.choice(CITIES)} and requested a temporary badge.",
            ),
            make_doc(
                3,
                "Building Memo",
                "Badge colors are used for routing employees through secure zones.",
            ),
        ]

        question = f"What color is {person}'s access badge?"
        answer = color

    elif kind == "cross_doc_join":
        person = synthetic_name(r)
        project = synthetic_word(r, 2).capitalize()
        department = r.choice(DEPARTMENTS)
        manager = synthetic_name(r)

        documents = [
            make_doc(
                1,
                "Project Assignment",
                f"{person} is assigned to Project {project}.",
            ),
            make_doc(
                2,
                "Project Directory",
                f"Project {project} belongs to the {department} department.",
            ),
            make_doc(
                3,
                "Department Roster",
                f"The {department} department is managed by {manager}.",
            ),
        ]

        question = f"Who manages the department for the project assigned to {person}?"
        answer = manager

    elif kind == "latest_fact_wins":
        person = synthetic_name(r)
        old_city = r.choice(CITIES)
        new_city = r.choice([city for city in CITIES if city != old_city])
        old_month_index = r.randint(0, 5)
        new_month_index = r.randint(old_month_index + 1, 11)

        documents = [
            make_doc(
                1,
                f"{MONTHS[old_month_index]} Directory",
                f"In {MONTHS[old_month_index]}, {person} was listed in {old_city}.",
            ),
            make_doc(
                2,
                f"{MONTHS[new_month_index]} Directory Update",
                f"In {MONTHS[new_month_index]}, {person} moved to {new_city}.",
            ),
            make_doc(
                3,
                "Archive Notice",
                "Older directory entries may be superseded by later updates.",
            ),
        ]

        question = f"According to the latest document, where is {person} located?"
        answer = new_city

    elif kind == "count_matching_docs":
        target_department = r.choice(DEPARTMENTS)
        people = [synthetic_name(r) for _ in range(5)]

        other_departments = [
            department for department in DEPARTMENTS if department != target_department
        ]

        assigned = [
            target_department,
            target_department,
            r.choice(other_departments),
            target_department,
            r.choice(other_departments),
        ]

        r.shuffle(assigned)

        documents = []

        for i, person in enumerate(people, start=1):
            documents.append(
                make_doc(
                    i,
                    f"Employee Record {i}",
                    f"{person} works in the {assigned[i - 1]} department.",
                )
            )

        question = f"How many documents describe someone in the {target_department} department?"
        answer = str(sum(1 for department in assigned if department == target_department))

    elif kind == "numeric_max":
        warehouses = [
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
        ]

        counts = r.sample(range(15, 95), k=len(warehouses))
        max_index = counts.index(max(counts))

        documents = []

        for i, warehouse in enumerate(warehouses, start=1):
            documents.append(
                make_doc(
                    i,
                    f"Warehouse {warehouse}",
                    f"Warehouse {warehouse} has {counts[i - 1]} units in stock.",
                )
            )

        question = "Which warehouse has the highest number of units in stock?"
        answer = f"Warehouse {warehouses[max_index]}"

    elif kind == "numeric_min":
        routes = [
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
            synthetic_word(r, 2).capitalize(),
        ]

        minutes = r.sample(range(20, 120), k=len(routes))
        min_index = minutes.index(min(minutes))

        documents = []

        for i, route in enumerate(routes, start=1):
            documents.append(
                make_doc(
                    i,
                    f"Route {route}",
                    f"Route {route} takes {minutes[i - 1]} minutes to complete.",
                )
            )

        question = "Which route takes the fewest minutes to complete?"
        answer = f"Route {routes[min_index]}"

    elif kind == "set_intersection":
        person_a = synthetic_name(r)
        person_b = synthetic_name(r)

        shared_item = r.choice(ITEMS)
        remaining_items = [item for item in ITEMS if item != shared_item]
        a_only = r.choice(remaining_items)
        remaining_items = [
            item for item in remaining_items if item != a_only
        ]
        b_only = r.choice(remaining_items)

        documents = [
            make_doc(
                1,
                "Inventory Record A",
                f"{person_a} checked out a {shared_item}, a {a_only}, and a notebook.",
            ),
            make_doc(
                2,
                "Inventory Record B",
                f"{person_b} checked out a {shared_item}, a {b_only}, and a charger.",
            ),
            make_doc(
                3,
                "Inventory Policy",
                "Only equipment listed in checkout records should be considered.",
            ),
        ]

        question = f"What equipment item was checked out by both {person_a} and {person_b}?"
        answer = shared_item

    elif kind == "distractor_filter":
        target_person = synthetic_name(r)
        distractor_person = synthetic_name(r)

        target_item = r.choice(ITEMS)
        distractor_item = r.choice([item for item in ITEMS if item != target_item])

        target_city = r.choice(CITIES)
        distractor_city = r.choice([city for city in CITIES if city != target_city])

        documents = [
            make_doc(
                1,
                "Correct Site Record",
                f"At the {target_city} site, {target_person} received a {target_item}.",
            ),
            make_doc(
                2,
                "Different Person Record",
                f"At the {target_city} site, {distractor_person} received a {distractor_item}.",
            ),
            make_doc(
                3,
                "Different Site Record",
                f"At the {distractor_city} site, {target_person} received a {distractor_item}.",
            ),
        ]

        question = f"What item did {target_person} receive at the {target_city} site?"
        answer = target_item

    elif kind == "multi_hop_owner_item":
        person = synthetic_name(r)
        team = synthetic_word(r, 2).capitalize()
        locker = synthetic_word(r, 1).capitalize() + "-" + str(r.randint(10, 99))
        item = r.choice(ITEMS)

        documents = [
            make_doc(
                1,
                "Team Assignment",
                f"{person} belongs to Team {team}.",
            ),
            make_doc(
                2,
                "Locker Directory",
                f"Team {team} uses locker {locker}.",
            ),
            make_doc(
                3,
                "Locker Contents",
                f"Locker {locker} contains a {item}.",
            ),
            make_doc(
                4,
                "General Policy",
                "Teams may only access the locker assigned in the locker directory.",
            ),
        ]

        question = f"What item is in the locker used by {person}'s team?"
        answer = item

    elif kind == "date_order_lookup":
        person = synthetic_name(r)
        first_item = r.choice(ITEMS)
        second_item = r.choice([item for item in ITEMS if item != first_item])
        third_item = r.choice(
            [item for item in ITEMS if item not in {first_item, second_item}]
        )

        months = r.sample(range(12), k=3)
        months.sort()

        documents = [
            make_doc(
                1,
                f"{MONTHS[months[0]]} Checkout",
                f"In {MONTHS[months[0]]}, {person} checked out a {first_item}.",
            ),
            make_doc(
                2,
                f"{MONTHS[months[1]]} Checkout",
                f"In {MONTHS[months[1]]}, {person} checked out a {second_item}.",
            ),
            make_doc(
                3,
                f"{MONTHS[months[2]]} Checkout",
                f"In {MONTHS[months[2]]}, {person} checked out a {third_item}.",
            ),
        ]

        question = f"What item did {person} check out earliest?"
        answer = first_item

    else:
        raise ValueError(f"Unknown AXE kind: {kind}")

    prompt = make_prompt(
        documents=documents,
        question=question,
    )

    return make_record(
        kind=kind,
        index=index,
        prompt=prompt,
        completion=answer,
        answer=answer,
        documents=documents,
        question=question,
        seed=seed,
    )


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random((int(seed) ^ AXE_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in AXE_KINDS:
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


def write_messages_jsonl(records: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            message_record = record_to_messages(record)
            f.write(json.dumps(message_record, ensure_ascii=False) + "\n")


def verify_record(record: dict) -> tuple[bool, str]:
    """
    Local verification for generated gold records.

    This does not call a model. It verifies that:
    - completion is one non-empty line
    - completion exactly matches the gold answer after light normalization
    - required fields are present
    - prompt does not leak the final answer after the Answer: marker
    - documents are structurally valid
    """
    try:
        required = [
            "src",
            "kind",
            "task_id",
            "prompt",
            "completion",
            "answer",
            "documents",
            "question",
            "status",
            "seed",
        ]

        for key in required:
            if key not in record:
                raise KeyError(f"Missing required key: {key}")

        completion = record["completion"]
        answer = record["answer"]

        nonempty_lines = [line for line in completion.splitlines() if line.strip()]

        if len(nonempty_lines) != 1:
            raise ValueError(
                f"Completion must have exactly one non-empty line, got {len(nonempty_lines)}"
            )

        if normalize_for_match(completion) != normalize_for_match(answer):
            raise AssertionError(
                f"Completion does not match answer: {completion!r} != {answer!r}"
            )

        prompt = record["prompt"]

        if not prompt.endswith("Answer:"):
            raise AssertionError("Prompt must end with 'Answer:'")

        answer_suffix = prompt.split("Answer:", 1)[1]

        if answer_suffix.strip():
            raise AssertionError("Prompt appears to contain text after the Answer: marker")

        documents = record["documents"]

        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list")

        for doc in documents:
            for key in ["id", "title", "text"]:
                if key not in doc:
                    raise KeyError(f"Document missing required key: {key}")

            if not isinstance(doc["id"], int):
                raise TypeError("Document id must be an integer")

            if not isinstance(doc["title"], str) or not doc["title"].strip():
                raise TypeError("Document title must be a non-empty string")

            if not isinstance(doc["text"], str) or not doc["text"].strip():
                raise TypeError("Document text must be a non-empty string")

        return True, ""

    except Exception:
        return False, traceback.format_exc()


def verify_records(records: list[dict], max_failures: int = 10) -> bool:
    failures = []

    for record in records:
        ok, err = verify_record(record)

        if not ok:
            failures.append((record.get("task_id"), record.get("kind"), err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated multi-doc AXE training records passed checks.")
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
            "Build a deterministic multi-document AXE JSONL training database "
            "with one-line gold answers."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/multi_doc_axe_database_all_cases.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260502,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-per-kind",
        type=int,
        default=10,
        help="Number of AXE records to generate per task kind.",
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
        help="Run local verification that generated records are well formed.",
    )

    parser.add_argument(
        "--format",
        choices=["completion", "messages"],
        default="completion",
        help="Output format: prompt/completion or chat messages.",
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

    if args.format == "completion":
        write_jsonl(records, output_path, append=args.append)
    else:
        write_messages_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"AXE kinds: {len(AXE_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print(f"Format: {args.format}")

    if args.format == "completion":
        print("Completion format: one-line answer only")
    else:
        print("Messages format: user prompt + assistant gold answer")

    print_counts(records)


if __name__ == "__main__":
    main()
