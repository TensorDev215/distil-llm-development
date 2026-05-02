from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import CrossEntropyLoss

model_path = "./models/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

from datasets import load_dataset

test_dataset = load_dataset("json", data_files="./datasets/Capybara_json/test.jsonl")["train"]

processed = []

for item in test_dataset:
    last_user = [msg["content"] for msg in item["messages"] if msg["role"] == "user"][-1]
    last_assistant = [msg["content"] for msg in item["messages"] if msg["role"] == "assistant"][-1]
    processed.append({"prompt": last_user, "completion": last_assistant})

print(f"Total test prompts: {len(processed)}")

loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")
all_losses = []

for example in processed:
    prompt = example["prompt"]
    target = example["completion"]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    labels = tokenizer(target, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([inputs.input_ids, labels], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        target_start = inputs.input_ids.shape[1]

        ce_loss = loss_fn(logits[:, target_start - 1: -1, :].reshape(-1, logits.size(-1)), labels.reshape(-1))

        all_losses.append(ce_loss.item())

avg_ce = sum(all_losses) / len(all_losses)
print(f"Average cross-entropy loss: {avg_ce:.4f}")