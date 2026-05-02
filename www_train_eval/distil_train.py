import torch
from datasets import load_dataset
from trl.experimental.distillation import DistillationConfig, DistillationTrainer
import wandb
import sys
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# # Define the objective function for Optuna
# def objective(trial: optuna.Trial):
#     # Suggest hyperparameters using the trial object
# num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
# lmbda = trial.suggest_categorical("lmbda", [0, 0.5, 1])  # Fixed values for lmbda
# beta = trial.suggest_categorical("beta", [0.9, 0.5])  # Fixed values for beta
# learning_rate = trial.suggest_float("learning_rate", 3e-6, 1e-5, log=True)  # Learning rate

# Load a smaller dataset
dataset = load_dataset("json", data_files="/root/train/dataset/reasoning_dataset.jsonl", split="train")  # Use 3% of the dataset
dataset = dataset.map(
    lambda x: {"messages": [{"role": "user", "content": x["prompt"]},
                            {"role": "assistant", "content": x["completion"]}]},
    remove_columns=dataset.column_names
)

# Split the dataset into training and evaluation sets
train_size = int(0.95 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# Peft Configuration

lora_config = LoraConfig(
    r=128,
    lora_alpha=1280,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# Configure the distillation
config = DistillationConfig(
    output_dir="/root/train/distil_lora",
    num_train_epochs=2,
    bf16=True,
    save_strategy="epoch",
    save_only_model=True,
    logging_steps=1,
    warmup_steps=10,
    learning_rate=1e-6,  # Add learning rate to the config
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.5},
    temperature = 1.0,
    top_p=0.95,
    lmbda=1,
    beta=0.9,
    num_generations = 1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    reverse_kl_top_1_mode = "sampled",
    max_length = 10000,
    max_completion_length = 2048,
    loss_top_k = 1,  # Add loss_top_k to the config
    teacher_model_init_kwargs={"torch_dtype": "bfloat16"},
    # use_teacher_server=True,  # Enable teacher server for distributed training
    # teacher_model_server_url = "http://localhost:9001",
    report_to="wandb",

    # use_vllm=True,  # Enable vLLM for training
    # vllm_gpu_memory_utilization=0.3,
    # vllm_mode="colocate"
)

# Initialize the trainer with vllm
trainer = DistillationTrainer(
    model="/root/train/model/student_model",
    teacher_model="/root/train/model/teacher_model",
    args=config,
    peft_config = lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]

wandb.log({"eval_loss": eval_loss})  # Log evaluation loss to Weights & Biases

# Save the model
# trainer.save_model("../temp/final")

# torch.cuda.empty_cache()  # Clear GPU memory after each trial