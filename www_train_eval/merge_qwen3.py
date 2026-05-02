from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import Qwen3_5ForConditionalGeneration
from peft import PeftModel
import torch

BASE_MODEL = "./arc-format"
LORA_DIR = "./output/v28-20260131-043805/checkpoint-57000"
MERGED_DIR = "./merge/checkpoint-57000"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",  
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


model = PeftModel.from_pretrained(
    model,
    LORA_DIR,
)


merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    MERGED_DIR,
    # safe_serialization=True,
    max_shard_size="5GB"
)
tokenizer.save_pretrained(MERGED_DIR)