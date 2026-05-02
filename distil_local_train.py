import os
import torch
import wandb
from datasets import load_dataset
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# ---Config---
DATASET = "karpathy/climbmix-400b-shuffle"

TEACHER_MODEL = "./models/Qwen3.5-2B"
STUDENT_MODEL = "./models/Qwen3.5-0.8B"

BATCH_SIZE = 4
LR = 2e-5
MAX_LEN = 128

SAVE_EVERY = 100
MAX_STEPS = 1000

OUTPUT_DIR = "./checkpoints/student_kl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---W&B---
wandb.init(
    project="distillation",
    name="qwen-step-checkpoints",
)

# ---Dataset---
dataset = load_dataset(
    DATASET,
    split="train",
    data_files="shard_00000.parquet",
    streaming=True,
)


def format_prompt(text):
    """
    Dataset schema:
    {"text": "..."}
    """
    if text is None:
        return ""

    text = str(text).strip()

    if len(text) == 0:
        return ""

    return text


# ---Tokenizer---
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---Teacher---

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# ---Student---

device = "cuda" if torch.cuda.is_available() else "cpu"

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    torch_dtype=torch.bfloat16,
).to(device)

student.train()

optimizer = AdamW(student.parameters(), lr=LR)

# ---Kl Loss Helper---
def kl_loss(student_logits, teacher_logprobs, mask):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logprobs, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none"
    )

    kl = kl.sum(-1)

    kl = kl * mask

    return kl.sum() / mask.sum().clamp(min=1)

# ---Training Loop---
student.train()

step = 0

for batch in dataset.batch(BATCH_SIZE):
    texts = batch["text"]

    prompts = [format_prompt(x) for x in texts]
    prompts = [x for x in prompts if x != ""]

    if len(prompts) == 0:
        continue

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    attention_mask = inputs["attention_mask"].float()

    with torch.no_grad():
        teacher_outputs = teacher(**inputs)
        teacher_logits = teacher_outputs.logits

    student_outputs = student(**inputs)
    student_logits = student_outputs.logits

    mask = (inputs["input_ids"] != tokenizer.pad_token_id).float()

    loss = kl_loss(student_logits, teacher_logits, mask)


    # ---Backprop---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value = loss.item()

    # ---Logging---
    wandb.log({
        "kl_loss": loss_value,
        "step": step,
    })

    print(f"step={step} kl_loss={loss_value:.4f}")

    # ---Save every N steps---
    if step % SAVE_EVERY == 0 and step > 0:
        ckpt_path = os.path.join(
            OUTPUT_DIR,
            f"checkpoint_step_{step}"
        )

        student.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

        print(f"Saved checkpoint at {ckpt_path}")


        wandb.log({
            "checkpoint_saved": step
        })

    step += 1

    # ---Stop condition---
    if step >= MAX_STEPS:
        break

# ---Final save---
final_path = os.path.join(OUTPUT_DIR, "final")
student.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

wandb.finish()

print(f"\nFinished training. Final model saved to: {final_path}")