from datasets import load_dataset, interleave_datasets
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# from peft import LoraConfig

MODEL_PATH = "/home/shadeform/affine/sft_model_lgc_finetune_full_v1/checkpoint-60738"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="./dataset_lgc_0_80978_total.jsonl")["train"]

def preprocess(example, index):
    question = example["question"]
    answer = example["answer"]

    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }

dataset = dataset.map(preprocess, with_indices=True, remove_columns=["question", "answer", "index"])

import time

# dataset = dataset.shuffle(seed=int(time.time()))

training_args = SFTConfig(
    output_dir="./sft_model_lgc_finetune_full_v3",
    num_train_epochs=30,
    max_length=36500,
    # packing=True,
    # assistant_only_loss=True,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=4,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.5},
    # load_best_model_at_end=True,
    # eval_strategy="steps",
    save_strategy="epoch",
    # eval_steps=5,
    # save_steps=1000,
    learning_rate=5e-5,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_only_model=True,
    warmup_ratio=0.0005,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(20)),
    args=training_args,
)

trainer.train()