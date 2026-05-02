from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch
import json
import os
import shutil

BASE_MODEL = "./winner"
LORA_DIR = "./distilled_model/checkpoint-2500"
MERGED_DIR = "./merge/checkpoint_2500"

os.makedirs(MERGED_DIR, exist_ok=True)

# ─── Step 1: Merge LoRA into text backbone (correct architecture) ───
print("Loading base model (text backbone)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Applying LoRA...")
model = PeftModel.from_pretrained(model, LORA_DIR)

print("Merging weights...")
merged_model = model.merge_and_unload()

print("Saving merged weights...")
merged_model.save_pretrained(MERGED_DIR, max_shard_size="5GB")

# ─── Step 2: Save processor (not just tokenizer) ───
print("Saving processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(MERGED_DIR)

# ─── Step 3: Restore original VLM config.json ───
print("Restoring base VLM config...")
shutil.copy(
    os.path.join(BASE_MODEL, "config.json"),
    os.path.join(MERGED_DIR, "config.json")
)

# ─── Step 4: Copy vision weights from base model ───
# The merged model only has text weights (model.* keys)
# We need to also copy the visual.* weights from the base model
print("Copying vision weights from base model...")

from safetensors.torch import load_file, save_file
from collections import defaultdict
import glob

# Load all base model shards and extract visual.* weights
visual_weights = {}
base_shards = sorted(glob.glob(os.path.join(BASE_MODEL, "*.safetensors")))

for shard_path in base_shards:
    print(f"  Scanning {os.path.basename(shard_path)}...")
    shard = load_file(shard_path)
    for key, tensor in shard.items():
        if key.startswith("visual.") or key.startswith("language_model."):
            # We only need visual.* — language_model.* comes from merged text weights
            if key.startswith("visual."):
                visual_weights[key] = tensor

print(f"  Found {len(visual_weights)} visual weight tensors")

# Save visual weights as a separate shard
if visual_weights:
    visual_shard_path = os.path.join(MERGED_DIR, "visual_weights.safetensors")
    save_file(visual_weights, visual_shard_path)
    print(f"  Saved visual weights to visual_weights.safetensors")

    # Update model.safetensors.index.json to include visual shard
    index_path = os.path.join(MERGED_DIR, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)

        for key in visual_weights:
            index["weight_map"][key] = "visual_weights.safetensors"

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print("  Updated model.safetensors.index.json")

print("\n✅ Merge complete!")

# ─── Step 5: Verify ───
with open(os.path.join(MERGED_DIR, "config.json")) as f:
    cfg = json.load(f)

print(f"  architecture : {cfg.get('architectures')}")
print(f"  model_type   : {cfg.get('model_type')}")
print(f"  vision_config: {'✅ present' if 'vision_config' in cfg else '❌ missing'}")
print(f"  image_token_id: {cfg.get('image_token_id')}")
