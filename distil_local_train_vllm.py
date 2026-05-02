import os
import torch
import wandb
from datasets import load_from_disk
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

# ---Config---
DATASET = "karpathy/climbmix-400b-shuffle"

TEACHER_MODEL = "./models/Qwen3.5-2B"
STUDENT_MODEL = "./models/Qwen3.5-0.8B"

BATCH_SIZE = 8
LR = 2e-5
MAX_TOKENS = 128

SAVE_EVERY = 20
MAX_STEPS = 1000
OUTPUT_DIR = "./checkpoints/student_kl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---W&B---
wandb.init(
    project="vllm-distillation",
    name="qwen-step-checkpoints",
)

# ---Dataset---
dataset = load_from_disk("./local_dataset")

def format_prompt(x):
    return x["question"]

teacher = LLM(
    model=TEACHER_MODEL,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS,
    logprobs=20,
)

# ---Student---
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

optimizer = AdamW(student.parameters(), lr=LR)

# ---Kl Loss Helper---
def kl_loss(student_logits, teacher_logprobs, labels):
    """
    student_logits: [B, T, V]
    teacher_logprobs: list of token logprobs from vLLM
    """
    log_probs = F.log_softmax(student_logits, dim=-1)

    loss = 0.0
    count = 0

    for b in range(len(labels)):
        for t in range(labels[b].shape[0]):

            token_id = labels[b][t]

            if token_id == tokenizer.pad_token_id:
                continue

            student_logprob = log_probs[b, t, token_id]

            teacher_logprob = teacher_logprobs[b][t]

            loss += (teacher_logprob - student_logprob) ** 2
            count += 1

        return loss / max(count, 1)

# ---Training Loop---
student.train()

step = 0

for i in range(0, len(dataset), BATCH_SIZE):
    batch = dataset[i:i+BATCH_SIZE]
    prompts = [format_prompt(x) for x in batch]

    # ---Teacher inference (vLLM)---
    teacher_outputs = teacher.generate(prompts, sampling_params)

    teacher_texts = []
    teacher_logprobs = []

    for o in teacher_outputs:
        token_logprobs = []

        for tok in o.outputs[0].logprobs:
            top_token = list(tok.values())[0]
            token_logprobs.append(top_token.logprob)

        teacher_texts.append(o.outputs[0].text)
        teacher_logprobs.append(token_logprobs)

    # ---Tokenize---
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(student.device)

    labels = tokenizer(
        teacher_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids.to(student.device)

    # ---Forward---
    outputs = student(**inputs)

    logits = outputs.logits

    loss = kl_loss(logits, teacher_logprobs, labels)

    # ---Backprop---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---Logging---
    wandb.log({
        "kl_loss": loss.item(),
        "step": step,
    })

    print(f"step={step} kl_loss={loss.item():.4f}")

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
tokenizer.save_pretrained()

wandb.finish()