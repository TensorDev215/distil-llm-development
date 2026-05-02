from datasets import load_dataset

dataset_dict = load_dataset(
    "trl-lib/Capybara",
    split=None,              
)

import os

output_dir = "./datasets/Capybara_json"
os.makedirs(output_dir, exist_ok=True)

for split_name, ds in dataset_dict.items():
    out_file = os.path.join(output_dir, f"{split_name}.jsonl")
    ds.to_json(out_file)
    print(f"Saved {split_name} split to {out_file}")


