from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial

# -------------------------------
# Paths
# -------------------------------
merged_model_dir = "./sft_lora_qwen3/merged_model"
test_dataset_path = "./datasets/Capybara_json/test.jsonl"

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load tokenizer and merged model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
model = AutoModelForCausalLM.from_pretrained(merged_model_dir, trust_remote_code=True)
model.to(device)
model.eval()

# -------------------------------
# Load and preprocess test dataset
# -------------------------------
raw_test_dataset = load_dataset("json", data_files=test_dataset_path)["train"]

def preprocess(example):
    last_user = [msg["content"] for msg in example["messages"] if msg["role"] == "user"][-1]
    last_assistant = [msg["content"] for msg in example["messages"] if msg["role"] == "assistant"][-1]
    return {"prompt": last_user, "completion": last_assistant}

processed_dataset = [preprocess(x) for x in raw_test_dataset]
print(f"Total test prompts: {len(processed_dataset)}")

# -------------------------------
# Evaluation
# -------------------------------
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")
all_losses = []

batch_size = 2  # adjust based on GPU memory

for i in range(0, len(processed_dataset), batch_size):
    batch = processed_dataset[i:i+batch_size]

    input_ids_list = []
    label_ids_list = []

    for example in batch:
        prompt_ids = tokenizer(example["prompt"], return_tensors="pt").input_ids
        target_ids = tokenizer(example["completion"], return_tensors="pt").input_ids

        # concatenate prompt + target
        input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        input_ids_list.append(input_ids)
        label_ids_list.append(target_ids)

    # pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels_padded = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids=input_ids_padded, attention_mask=attention_mask)
        logits = outputs.logits

        for j in range(len(batch)):
            target_start = input_ids_list[j].shape[1] - label_ids_list[j].shape[1]
            # compute loss for target tokens only
            ce_loss = loss_fn(
                logits[j, target_start:-1, :].reshape(-1, logits.size(-1)),
                labels_padded[j].reshape(-1)
            )
            all_losses.append(ce_loss.item())

# Average cross-entropy loss
avg_ce = sum(all_losses) / len(all_losses)
print(f"Average cross-entropy loss on test set: {avg_ce:.4f}")