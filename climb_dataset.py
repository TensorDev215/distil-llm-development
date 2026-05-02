from datasets import load_dataset

dataset = load_dataset("karpathy/climbmix-400b-shuffle", split="train")
print(dataset.num_shards if hasattr(dataset, "num_shards") else "Shards info not directly exposed")