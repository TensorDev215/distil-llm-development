import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

#---Config---
DATABASE = "./local_dataset/00001"
TEACHER_MODEL = "./models/Qwen3.5-2B"
CHECKPOINT = "./models/Qwen3.5-2B"
# CHECKPOINT = "./checkpoints/student_kl/checkpoint_step_100"


BATCH_SIZE = 4
MAX_LEN = 128
MAX_EVAL_STEPS = 100

#---Load dataset---
dataset = load_from_disk(DATABASE)

print("Dataset columns:", dataset.column_names)
print("Dataset size:", len(dataset))

#---Tokenizer---
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

#---Teacher---
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.bfloat16
).to("cuda")

teacher.eval()

#---Student Checkpoint---
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.bfloat16
).to("cuda")

student.eval()

#---Forward KL---
# KL(teacher || student)
@torch.no_grad()
def forward_kl(student_logits, teacher_logits, mask):
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none"
    )

    kl = kl.sum(dim=-1)
    kl = kl * mask

    return kl.sum() / mask.sum().clamp(min=1)

#---Reverse KL---
#---KL(student || teacher)---
@torch.no_grad()
def reverse_kl(student_logits, teacher_logits, mask):
    student_probs = F.softmax(student_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)

    teacher_log_probs = F.log_softmax(
        teacher_logits.float(),
        dim=-1
    )

    rkl = student_probs * (
        student_log_probs - teacher_log_probs
    )

    rkl = rkl.sum(dim=-1)
    rkl = rkl * mask

    return rkl.sum() / mask.sum().clamp(min=1)

#---Eval Loop---
total_fkl = 0.0
total_rkl = 0.0
count=0

num_samples = min(
    len(dataset),
    BATCH_SIZE * MAX_EVAL_STEPS
)

print("\nStarting evaluation...\n")

for i in range(0, num_samples, BATCH_SIZE):

    batch = dataset[i:i + BATCH_SIZE]

    prompts = batch["text"]

    if len(prompts) == 0:
        continue

    print("=" * 60)
    print("Sample prompt:")
    print(prompts[0][:150].replace("\n", " "))
    print("=" * 60)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to("cuda")

    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits
        student_logits = student(**inputs).logits

    mask = (
        inputs["input_ids"] != tokenizer.pad_token_id
    ).float()

    fkl = forward_kl(
        student_logits,
        teacher_logits,
        mask
    )

    rkl = reverse_kl(
        student_logits,
        teacher_logits,
        mask
    )

    total_fkl += fkl.item()
    total_rkl += rkl.item()
    count += 1

    print(
        f"batch={count} | "
        f"FKL={fkl.item():.6f} | "
        f"RKL={rkl.item():.6f}"
    )


#---Final result---
print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Average Forward KL : {total_fkl / count:.6f}")
print(f"Average Reverse KL : {total_rkl / count:.6f}")
print("=" * 60)