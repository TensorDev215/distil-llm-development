import os
import torch
import wandb
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from sklearn.model_selection import train_test_split

# =========================
# Paths and basic settings
# =========================
model_path = "./models/Qwen3-0.6B"
train_json1 = "./datasets/Capybara_json/train.jsonl"
output_dir = "./sft_lora_qwen3"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Initialize Weights & Biases
# =========================
wandb.init(
    project="qwen3-sft",
    name="qwen3-0.6b-lora-run1",
    mode="online",  # ensures browser sync
    config={
        "model_path": model_path,
        "train_file": train_json1,
        "num_train_epochs": 3,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "max_length": 512,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "eval_split": 0.1,
    }
)

# =========================
# Load dataset
# =========================
raw_dataset = load_dataset("json", data_files={"train": train_json1})["train"]

def preprocess(example):
    user_messages = [msg["content"] for msg in example["messages"] if msg["role"] == "user"]
    assistant_messages = [msg["content"] for msg in example["messages"] if msg["role"] == "assistant"]

    if len(user_messages) == 0 or len(assistant_messages) == 0:
        return {"prompt": "", "completion": ""}

    last_user = user_messages[-1]
    last_assistant = assistant_messages[-1]
    return {"prompt": last_user, "completion": last_assistant}

dataset = raw_dataset.map(preprocess)

# Remove empty rows just in case
dataset = dataset.filter(lambda x: x["prompt"].strip() != "" and x["completion"].strip() != "")

# =========================
# Train / validation split
# =========================
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.1,
    random_state=42
)

train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

print(f"Number of dataset examples: {len(dataset)}")
print(f"Number of train dataset examples: {len(train_dataset)}")
print(f"Number of validation dataset examples: {len(val_dataset)}")

# =========================
# LoRA config
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# =========================
# Load tokenizer and model
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)

model.to(device)

# =========================
# Training config
# =========================
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="steps",
    save_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=5,
    report_to="wandb",
    run_name="qwen3-0.6b-lora",
    save_total_limit=2,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()),
    logging_dir=os.path.join(output_dir, "logs"),
)

# =========================
# Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    args=training_args
)

# =========================
# Train
# =========================
train_result = trainer.train()

# Save final trainer state metrics too
trainer.save_state()

# =========================
# Collect loss history
# =========================
log_history = trainer.state.log_history

train_steps = []
train_losses = []
eval_steps = []
eval_losses = []

for log in log_history:
    if "loss" in log and "epoch" in log:
        train_steps.append(log.get("step", len(train_steps)))
        train_losses.append(log["loss"])

    if "eval_loss" in log:
        eval_steps.append(log.get("step", len(eval_steps)))
        eval_losses.append(log["eval_loss"])

# =========================
# Plot loss curves
# =========================
plt.figure(figsize=(8, 5))

if len(train_losses) > 0:
    plt.plot(train_steps, train_losses, label="Train Loss")

if len(eval_losses) > 0:
    plt.plot(eval_steps, eval_losses, label="Validation Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plot_path = os.path.join(output_dir, "loss_plot.png")
plt.savefig(plot_path)
plt.close()

# Log plot to wandb
wandb.log({"loss_plot": wandb.Image(plot_path)})

# =========================
# Save LoRA adapter
# =========================
adapter_dir = os.path.join(output_dir, "lora_adapter")
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

# Optional: log adapter as wandb artifact
artifact = wandb.Artifact("qwen3-lora-adapter", type="model")
artifact.add_dir(adapter_dir)
wandb.log_artifact(artifact)

print(f"Training finished. LoRA adapter saved to {adapter_dir}")
print(f"Loss plot saved to {plot_path}")

# =========================
# Finish wandb run
# =========================
wandb.finish()
