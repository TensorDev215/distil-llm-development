import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distil.scripts.pod_eval_vllm as eval_node


# ── tool_use_bench helpers ─────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)

_TOOL_USE_INSTRUCTION = (
    "You have access to a Python calculator. To use it, write "
    "`<python>CODE</python>` and the environment will execute CODE and "
    "return the stdout as `<output>RESULT</output>`. Then continue and "
    "give your final answer inside `\\boxed{ANSWER}`. "
    "If you don't need the calculator, just solve normally."
)


def _tool_use_run_sandboxed(code: str, timeout_s: float) -> str:
    """
    Execute code in an isolated subprocess and return captured stdout.
    """
    if not code.strip():
        return ""

    import subprocess
    import tempfile
    import os as _os
    import sys as _sys

    try:
        with tempfile.TemporaryDirectory(prefix="tool_use_") as tmp:
            script = _os.path.join(tmp, "snippet.py")

            with open(script, "w", encoding="utf-8") as f:
                f.write(code)

            env = {
                "PATH": _os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "HOME": tmp,
                "TMPDIR": tmp,
            }

            try:
                proc = subprocess.run(
                    [_sys.executable, "-I", "-S", script],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )

                out_s = (proc.stdout or "")[:400]

                if not out_s and proc.stderr:
                    out_s = f"[stderr] {(proc.stderr or '')[:400]}"

                return out_s

            except subprocess.TimeoutExpired:
                return "[timeout]"

            except Exception as e:
                return f"[error] {str(e)[:200]}"

    except Exception as e:
        return f"[sandbox-err] {str(e)[:200]}"


# IMPORTANT:
# Keep your existing _generate_math_items(...) implementation here.
# The script below expects this function to exist:
#


# ── config ──────────────────────────────────────────────────────────────

BLOCK_SEED = 20260418
PROC_COUNT = 50
OUTPUT_FILE = "./dataset/tool_use_dataset.jsonl"

MODEL_PATH = "models/Qwen3.5-2B"
DEVICE = "cuda"
DTYPE = "bfloat16"

MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0
TOP_P = 1.0

BENCH_TOOL_USE_SANDBOX_TIMEOUT_S = 4.0


# ── args ────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate tool-use completions without vLLM and write all examples to JSONL."
    )

    parser.add_argument("--block-seed", type=int, default=BLOCK_SEED)
    parser.add_argument("--proc-count", type=int, default=PROC_COUNT)
    parser.add_argument("--output", type=str, default=OUTPUT_FILE)

    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE)

    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)

    parser.add_argument(
        "--sandbox-timeout",
        type=float,
        default=BENCH_TOOL_USE_SANDBOX_TIMEOUT_S,
    )

    return parser


# ── model loading ───────────────────────────────────────────────────────

def get_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()

    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if dtype_name in {"fp16", "float16"}:
        return torch.float16

    if dtype_name in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype_name: str,
):
    torch_dtype = get_torch_dtype(dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer


# ── generation ──────────────────────────────────────────────────────────

def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int]:
    """
    Generate text directly from a local HF model.

    Returns:
        generated_text, generated_token_count
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return text.strip(), int(new_ids.shape[-1])


# ── dataset building ────────────────────────────────────────────────────

def build_prompts(block_seed: int, proc_count: int) -> list[dict]:
    """
    Build prompts for every generated math item.

    No numeric filtering.
    No correctness filtering.
    """
    sampled = eval_node._generate_math_items(block_seed ^ 0x546F, proc_count)

    prompts = []

    for item in sampled:
        question = item.get("question", "").strip()

        if not question:
            continue

        prompt = f"{question}\n\n{_TOOL_USE_INSTRUCTION}"

        prompts.append(
            {
                "src": item.get("src", ""),
                "question": question,
                "gold": item.get("gold", ""),
                "prompt": prompt,
            }
        )

    return prompts


def process_item(
    item: dict,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sandbox_timeout: float,
) -> dict:
    """
    Generate one full tool-use trajectory.

    This writes every example:
    - correct or wrong,
    - tool used or not,
    - second pass successful or not.

    No scoring is done.
    """
    record = {
        "src": item.get("src", ""),
        "question": item.get("question", ""),
        "gold": item.get("gold", ""),
        "prompt": item.get("prompt", ""),
        "completion": "",
        "used_tool": False,
        "tool_call": None,
        "tool_result": None,
        "gen_tokens": 0,
        "status": "unknown",
        "error": None,
    }

    try:
        text1, tok1 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        record["gen_tokens"] += tok1

        if not text1:
            record["status"] = "empty_first_response"
            return record

        combined_text = text1

        tool_match = _TOOL_CALL_RE.search(text1)

        if not tool_match:
            record["completion"] = combined_text
            record["status"] = "generated_without_tool"
            return record

        record["used_tool"] = True

        code = tool_match.group(1)
        record["tool_call"] = code

        tool_result = _tool_use_run_sandboxed(
            code,
            sandbox_timeout,
        )

        record["tool_result"] = tool_result

        pass2_prompt = (
            f"{item['prompt']}\n"
            f"{text1[:tool_match.end()]}\n"
            f"<output>{tool_result}</output>\n"
            "Based on the tool output, give your final answer in \\boxed{ANSWER}."
        )

        text2, tok2 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=pass2_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        record["gen_tokens"] += tok2

        combined_text = (
            text1[:tool_match.end()]
            + f"\n<output>{tool_result}</output>\n"
            + text2
        )

        record["completion"] = combined_text

        if not text2:
            record["status"] = "empty_second_response"
        else:
            record["status"] = "generated_with_tool"

        return record

    except Exception as exc:
        record["status"] = "generation_failed"
        record["error"] = str(exc)[:500]
        return record


def write_jsonl_record(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def build_with_completions(
    block_seed: int,
    proc_count: int,
    output_path: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sandbox_timeout: float,
) -> int:
    prompts = build_prompts(
        block_seed=block_seed,
        proc_count=proc_count,
    )

    written = 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(prompts, desc="Generating tool-use completions"):
            record = process_item(
                item=item,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                sandbox_timeout=sandbox_timeout,
            )

            write_jsonl_record(f, record)
            written += 1

    print(f"[gen] Wrote {written} items -> {output_path}")
    return written


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        device=args.device,
        dtype_name=args.dtype,
    )

    build_with_completions(
        block_seed=args.block_seed,
        proc_count=args.proc_count,
        output_path=args.output,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sandbox_timeout=args.sandbox_timeout,
    )


if __name__ == "__main__":
    main()
