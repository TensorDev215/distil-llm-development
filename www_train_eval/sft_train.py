import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# ======================
# Configuration
# ======================
MODEL_PATH = "./model/johngreend"
DATA_PATH = "./dataset/dyck_error_sft_train.jsonl"


OUTPUT_LORA_DIR = "./lora/johngreend/"

torch.backends.cuda.matmul.allow_tf32 = True

# ======================
# Load Model
# ======================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=None,
)
model.config.use_cache = False

# Enable gradient checkpointing to reduce VRAM
model.gradient_checkpointing_enable()

# ======================
# Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ======================
# Dataset
# ======================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.shuffle(seed = 40)
def preprocess(example):
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "completion": [{"role": "assistant", "content": example["answer"]}],
    }

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)
# dataset.to_json("./dataset/train_grpo.jsonl")


# ======================
# Data Filtering by Token Length
# ======================
# MAX_TOKENS = 40000
# def filter_by_token_length(example):
#     # Concatenate all message contents
#     text = " ".join(
#         msg["content"]
#         for msg in example["messages"]
#         if "content" in msg and msg["content"] is not None
#     )

#     token_count = len(
#         tokenizer(
#             text,
#             add_special_tokens=False,
#             truncation=False
#         )["input_ids"]
#     )

#     return token_count < MAX_TOKENS
# dataset = dataset.filter(filter_by_token_length)

# ======================
# LoRA Configuration
# ======================
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# Training Arguments
# ======================
training_args = SFTConfig(
    output_dir=OUTPUT_LORA_DIR,
    num_train_epochs=5,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,

    learning_rate=3e-6,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.5},
    warmup_steps=10,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    save_only_model=True,
    max_length=None,
    optim="adamw_torch_fused",
    shuffle_dataset=True,
    # assistant_only_loss=True,
    use_liger_kernel=True,
)

# ======================
# Trainer
# ======================
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    # eval_dataset=dataset.select(range(min(20, len(dataset)))),
    args=training_args,
)

trainer.train(resume_from_checkpoint=False)