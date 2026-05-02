import optuna
import torch
from datasets import load_dataset
from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from transformers import AutoTokenizer
import wandb
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import sys

# Add the eval node to path if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import distil.scripts.pod_eval_vllm as eval_node

REPO_SCRIPT = REPO_ROOT / "distil" / "scripts"
sys.path.insert(1, str(REPO_SCRIPT))


# ═══════════════════════════════════════════════════════════════════
# §1  Eval Benchmark Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CustomEvalConfig:
    """
    Hyperparameter-driven benchmark selector.

    Pass this to custom_eval() to control which axes run.
    Each flag maps to one probe from pod_eval_vllm.py.

    Example — run only ifeval + capability:
        cfg = CustomEvalConfig(
            run_ifeval=True,
            run_capability=True,
            block_seed=42,
        )
    """
    # ── Benchmark axes ──────────────────────────────────────────────
    run_ifeval: bool = False            # IFEval instruction-following
    run_math: bool = False              # GSM8K + MATH-500 (procedural)
    run_code: bool = False              # HumanEval-style coding
    run_reasoning: bool = False         # BBH-style reasoning MC
    run_knowledge: bool = False         # MMLU-Pro style MC
    run_aime: bool = False              # Olympiad math (AIME-style)
    run_mbpp: bool = False              # MBPP+ programming
    run_long_context: bool = False      # Needle-in-haystack retrieval
    run_procedural: bool = False        # Synthetic procedural tasks
    run_robustness: bool = False        # Paraphrase robustness
    run_arc: bool = False               # ARC-Challenge science MC
    run_truthful: bool = False          # TruthfulQA adversarial MC

    # ── Probe axes (model behavior) ─────────────────────────────────
    run_capability: bool = True         # Verifiable short-answer battery
    run_chat_probe: bool = False        # Chat termination / collapse probe
    run_think_probe: bool = False       # CoT thinking collapse probe
    run_finetune_probe: bool = False    # Fine-tunability / anti-watermark

    # ── Sampling config ─────────────────────────────────────────────
    block_seed: Optional[int] = None    # Rotates prompt sets per round
    device: str = "cuda"

    # ── Per-bench token budgets (override defaults) ──────────────────
    math_max_tokens: Optional[int] = None
    code_max_tokens: Optional[int] = None
    reasoning_max_tokens: Optional[int] = None
    ifeval_max_tokens: Optional[int] = None
    aime_max_tokens: Optional[int] = None
    capability_max_tokens: Optional[int] = None

    # ── Per-bench sample counts (override defaults) ──────────────────
    math_per_round: Optional[int] = None
    code_per_round: Optional[int] = None
    reasoning_per_round: Optional[int] = None
    ifeval_per_round: Optional[int] = None
    aime_per_round: Optional[int] = None
    capability_n: Optional[int] = None
    capability_n_proc: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════
# §2  Custom Eval Entry Point
# ═══════════════════════════════════════════════════════════════════

def custom_eval(
    model,
    tokenizer,
    cfg: CustomEvalConfig,
    step: Optional[int] = None,
    log_to_wandb: bool = True,
) -> dict:
    """
    Run selected benchmark probes on a loaded model.

    Args:
        model:          A loaded HuggingFace CausalLM (already on device).
        tokenizer:      Matching tokenizer.
        cfg:            CustomEvalConfig controlling which axes run.
        step:           Current training step (for W&B x-axis).
        log_to_wandb:   Whether to push results to W&B.

    Returns:
        dict of {axis_name: pass_frac_or_score} for all axes that ran.
    """
    device = cfg.device
    results = {}

    # ── Apply any per-bench overrides ────────────────────────────────
    _apply_eval_overrides(cfg)

    # ── Seed the per-round prompt pools ─────────────────────────────
    if cfg.block_seed is not None:
        eval_node.set_capability_block_seed(cfg.block_seed)
        eval_node.set_bench_block_seed(cfg.block_seed)
        eval_node.set_judge_probe_block_seed(cfg.block_seed)
        eval_node.set_chat_turns_probe_block_seed(cfg.block_seed)
        eval_node.set_on_policy_rkl_block_seed(cfg.block_seed)

    print(f"\n{'='*55}", flush=True)
    print(f"[custom_eval] Starting benchmark evaluation", flush=True)
    if step is not None:
        print(f"[custom_eval] Training step: {step}", flush=True)
    print(f"{'='*55}", flush=True)

    # ── Probe axes ───────────────────────────────────────────────────

    if cfg.run_finetune_probe:
        results.update(_run_finetune_probe(model, tokenizer, device, cfg))

    if cfg.run_chat_probe:
        results.update(_run_chat_probe(model, tokenizer, device))

    if cfg.run_think_probe:
        results.update(_run_think_probe(model, tokenizer, device))

    if cfg.run_capability:
        results.update(_run_capability(model, tokenizer, device))

    # ── Benchmark axes ───────────────────────────────────────────────

    if cfg.run_math:
        results.update(_run_bench("math", eval_node.math_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_code:
        results.update(_run_bench("code", eval_node.code_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_reasoning:
        results.update(_run_bench("reasoning", eval_node.reasoning_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_knowledge:
        results.update(_run_bench("knowledge", eval_node.knowledge_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_ifeval:
        results.update(_run_bench("ifeval", eval_node.ifeval_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_aime:
        results.update(_run_bench("aime", eval_node.aime_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_mbpp:
        results.update(_run_bench("mbpp", eval_node.mbpp_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_long_context:
        results.update(_run_bench("long_context", eval_node.long_context_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_procedural:
        results.update(_run_bench("procedural", eval_node.procedural_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_robustness:
        results.update(_run_bench("robustness", eval_node.robustness_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_arc:
        results.update(_run_bench("arc", eval_node.arc_bench_probe,
                                  model, tokenizer, device))

    if cfg.run_truthful:
        results.update(_run_bench("truthful", eval_node.truthful_bench_probe,
                                  model, tokenizer, device))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n[custom_eval] Results summary:", flush=True)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)

    # ── W&B logging ──────────────────────────────────────────────────
    if log_to_wandb:
        try:
            wb_payload = {f"eval/{k}": v for k, v in results.items()
                          if isinstance(v, (int, float))}
            if step is not None:
                wandb.log(wb_payload, step=step)
            else:
                wandb.log(wb_payload)
            print(f"[custom_eval] Logged {len(wb_payload)} metrics to W&B", flush=True)
        except Exception as e:
            print(f"[custom_eval] W&B logging failed (non-fatal): {e}", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════════
# §3  Internal Probe Runners
# ═══════════════════════════════════════════════════════════════════

def _apply_eval_overrides(cfg: CustomEvalConfig):
    """Push per-bench overrides from cfg into the eval_node globals."""
    overrides = {
        "BENCH_MATH_PER_ROUND":       cfg.math_per_round,
        "BENCH_CODE_PER_ROUND":       cfg.code_per_round,
        "BENCH_REASONING_PER_ROUND":  cfg.reasoning_per_round,
        "BENCH_IFEVAL_PER_ROUND":     cfg.ifeval_per_round,
        "BENCH_AIME_PER_ROUND":       cfg.aime_per_round,
        "CAPABILITY_PROBE_N":         cfg.capability_n,
        "CAPABILITY_PROBE_N_PROC_MATH": cfg.capability_n_proc,
        "BENCH_MATH_MAX_TOKENS":      cfg.math_max_tokens,
        "BENCH_CODE_MAX_TOKENS":      cfg.code_max_tokens,
        "BENCH_REASONING_MAX_TOKENS": cfg.reasoning_max_tokens,
        "BENCH_IFEVAL_MAX_TOKENS":    cfg.ifeval_max_tokens,
        "BENCH_AIME_MAX_TOKENS":      cfg.aime_max_tokens,
        "CAPABILITY_PROBE_MAX_TOKENS": cfg.capability_max_tokens,
    }
    for attr, val in overrides.items():
        if val is not None and hasattr(eval_node, attr):
            setattr(eval_node, attr, val)


def _run_bench(name: str, probe_fn, model, tokenizer, device: str) -> dict:
    """Run a single bench probe and return a flat result dict."""
    print(f"\n[custom_eval] Running {name}_bench ...", flush=True)
    try:
        out = probe_fn(model, tokenizer, device)
        n = out.get("n", 0)
        correct = out.get("correct", 0)
        pass_frac = out.get("pass_frac", 0.0)
        mean_tok = out.get("mean_gen_tokens", 0.0)
        err = out.get("error")

        if err:
            print(f"  [{name}] error: {err[:120]}", flush=True)
            return {f"{name}_bench/pass_frac": 0.0,
                    f"{name}_bench/error": 1.0}

        print(f"  [{name}] {correct}/{n} = {pass_frac*100:.1f}%  "
              f"(mean_tokens={mean_tok:.0f})", flush=True)
        return {
            f"{name}_bench/pass_frac":        round(pass_frac, 4),
            f"{name}_bench/correct":          correct,
            f"{name}_bench/n":                n,
            f"{name}_bench/mean_gen_tokens":  round(mean_tok, 1),
        }
    except Exception as e:
        print(f"  [{name}] probe raised {type(e).__name__}: {e}", flush=True)
        return {f"{name}_bench/pass_frac": 0.0}


def _run_capability(model, tokenizer, device: str) -> dict:
    """Run the verifiable short-answer capability battery."""
    print(f"\n[custom_eval] Running capability probe ...", flush=True)
    try:
        out = eval_node.capability_probe(model, tokenizer, device)
        n = out.get("n", 0)
        correct = out.get("correct", 0)
        pass_frac = out.get("pass_frac", 0.0)
        print(f"  [capability] {correct}/{n} = {pass_frac*100:.1f}%", flush=True)
        return {
            "capability/pass_frac": round(pass_frac, 4),
            "capability/correct":   correct,
            "capability/n":         n,
        }
    except Exception as e:
        print(f"  [capability] error: {e}", flush=True)
        return {"capability/pass_frac": 0.0}


def _run_chat_probe(model, tokenizer, device: str) -> dict:
    """Run the chat termination / collapse probe."""
    print(f"\n[custom_eval] Running chat_probe ...", flush=True)
    try:
        out = eval_node.chat_response_probe(model, tokenizer, device)
        n = out.get("prompts_tested", 0)
        term = out.get("prompts_terminated", 0)
        term_rate = term / max(1, n)
        passed = int(out.get("pass", True))
        mean_tok = out.get("mean_gen_tokens", 0.0)
        think_frac = out.get("mean_reasoning_fraction", 0.0)
        print(f"  [chat_probe] terminated={term}/{n}  "
              f"mean_gen={mean_tok:.0f}  think_frac={think_frac:.2f}  "
              f"pass={bool(passed)}", flush=True)
        return {
            "chat_probe/termination_rate":    round(term_rate, 4),
            "chat_probe/mean_gen_tokens":     round(mean_tok, 1),
            "chat_probe/mean_think_fraction": round(think_frac, 3),
            "chat_probe/pass":                float(passed),
        }
    except Exception as e:
        print(f"  [chat_probe] error: {e}", flush=True)
        return {"chat_probe/pass": 0.0}


def _run_think_probe(model, tokenizer, device: str) -> dict:
    """Run the thinking-collapse degeneracy probe."""
    print(f"\n[custom_eval] Running think_probe ...", flush=True)
    try:
        out = eval_node.thinking_collapse_probe(model, tokenizer, device)
        n = out.get("prompts_tested", 0)
        term = out.get("prompts_terminated", 0)
        degen = out.get("prompts_degenerate", 0)
        sb = out.get("self_bleu_across_prompts", 0.0)
        passed = int(out.get("pass", True))
        print(f"  [think_probe] term={term}/{n}  degen={degen}/{n}  "
              f"self_bleu={sb:.3f}  pass={bool(passed)}", flush=True)
        return {
            "think_probe/termination_rate": round(term / max(1, n), 4),
            "think_probe/degen_rate":       round(degen / max(1, n), 4),
            "think_probe/self_bleu":        round(sb, 4),
            "think_probe/pass":             float(passed),
        }
    except Exception as e:
        print(f"  [think_probe] error: {e}", flush=True)
        return {"think_probe/pass": 0.0}


def _run_finetune_probe(model, tokenizer, device: str,
                        cfg: CustomEvalConfig) -> dict:
    """Run the fine-tunability / anti-watermark probe."""
    print(f"\n[custom_eval] Running finetune_probe ...", flush=True)
    try:
        out = eval_node.finetunability_probe(
            model, tokenizer, device, block_seed=cfg.block_seed)
        passed = int(out.get("pass", True))
        grad_norm = out.get("global_grad_norm", 0.0)
        norm_w = out.get("worst_norm_weight", 0.0)
        loss = out.get("loss", 0.0)
        print(f"  [finetune_probe] pass={bool(passed)}  "
              f"grad_norm={grad_norm:.1f}  norm_weight={norm_w:.2f}  "
              f"loss={loss:.4f}", flush=True)
        return {
            "finetune_probe/pass":             float(passed),
            "finetune_probe/global_grad_norm": round(grad_norm, 2),
            "finetune_probe/worst_norm_weight": round(norm_w, 4),
            "finetune_probe/loss":             round(loss, 4),
        }
    except Exception as e:
        print(f"  [finetune_probe] error: {e}", flush=True)
        return {"finetune_probe/pass": 0.0}


# ═══════════════════════════════════════════════════════════════════
# §4  Trainer Callback (optional — hooks into TRL training loop)
# ═══════════════════════════════════════════════════════════════════

from transformers import TrainerCallback

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_cfg: CustomEvalConfig, tokenizer, trainer_ref=None):
        self.eval_cfg = eval_cfg
        self.tokenizer = tokenizer
        self.trainer_ref = trainer_ref  # ← inject the trainer itself

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        live_model = self.trainer_ref.model if self.trainer_ref is not None else model
        if live_model is None:
            return
        
        if self.trainer_ref is not None:
            raw_model = self.trainer_ref.accelerator.unwrap_model(live_model)
        else:
            raw_model = getattr(live_model, "module", live_model)
        
        print(f"\n[CustomEvalCallback] Step {state.global_step}", flush=True)

        was_training = raw_model.training
        raw_model.eval()

        try:
            with torch.no_grad():
                custom_eval(
                    model=raw_model,
                    tokenizer=self.tokenizer,
                    cfg=self.eval_cfg,
                    step=state.global_step,
                    log_to_wandb=True,
                )
        except Exception as e:
            print(f"[CustomEvalCallback] custom_eval raised {type(e).__name__}: {e}", flush=True)
        finally:
            if was_training:
                raw_model.train()



# ═══════════════════════════════════════════════════════════════════
# §5  Training Script
# ═══════════════════════════════════════════════════════════════════

# ── Dataset ──────────────────────────────────────────────────────────
dataset = load_dataset("openai/gsm8k", "main", split="train[:10%]")
dataset = dataset.map(
    lambda x: {"messages": [
        {"role": "user",      "content": x["question"]},
        {"role": "assistant", "content": x["answer"]},
    ]},
    remove_columns=dataset.column_names,
)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset  = dataset.select(range(train_size, len(dataset)))

# ── Tokenizer (needed for the callback) ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/d/models/Qwen/Qwen3.5-0.8B", trust_remote_code=True
)

# ── Eval config — pick your axes here ────────────────────────────────
eval_cfg = CustomEvalConfig(
    run_ifeval=False,          # ← IFEval instruction-following
    run_capability=True,      # ← Short-answer verifiable battery
    run_math=True,            # ← Procedural math (GSM8K-style)
    run_chat_probe=True,      # ← Chat termination health check
    run_finetune_probe=False, # ← Anti-watermark check (slow)
    block_seed=42,            # ← Rotates prompt sets per round
    # Override sample counts for faster local runs:
    math_per_round=6,
    # ifeval_per_round=6,
    capability_n=8,
    capability_n_proc=8,
)

# ── Distillation config ───────────────────────────────────────────────
config = DistillationConfig(
    output_dir="../temp/ckpts",
    num_train_epochs=3,
    bf16=True,
    logging_steps=1,
    lmbda=0.5,
    beta=0.9,
    learning_rate=1e-6,
    teacher_model_init_kwargs={"torch_dtype": "bfloat16"},
    report_to="wandb",
    # use_vllm=True,
    # vllm_gpu_memory_utilization=0.5,
    # vllm_mode="colocate",
    eval_strategy="epoch",   # triggers on_evaluate callback
)

# ── Trainer ───────────────────────────────────────────────────────────
trainer = DistillationTrainer(
    model="/mnt/d/models/Qwen/Qwen3.5-0.8B",
    teacher_model="/mnt/d/models/Qwen/Qwen3.5-2B",
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ── Attach the custom eval callback ──────────────────────────────────
trainer.add_callback(
    CustomEvalCallback(eval_cfg=eval_cfg, tokenizer=tokenizer, trainer_ref=trainer)
)

# ── Train ─────────────────────────────────────────────────────────────
trainer.train()

# ── Final eval (standard TRL loss + custom axes) ─────────────────────
eval_results = trainer.evaluate()
wandb.log({"eval_loss": eval_results["eval_loss"]})

# ── Post-training custom eval on saved model ──────────────────────────
trainer.save_model("../temp/final")

# Run one final custom eval on the saved model
from transformers import AutoModelForCausalLM
final_model = AutoModelForCausalLM.from_pretrained(
    "../temp/final", torch_dtype=torch.bfloat16, device_map="cuda"
)
final_model.eval()

final_eval = custom_eval(
    model=final_model,
    tokenizer=tokenizer,
    cfg=eval_cfg,
    step=None,
    log_to_wandb=True,
)
print("\nFinal eval results:", final_eval)

torch.cuda.empty_cache()