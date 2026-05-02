from datasets import load_dataset

DATASET = "karpathy/climbmix-400b-shuffle"

dataset = load_dataset(
    DATASET,
    split="train",
    data_files="shard_00000.parquet"
)

dataset.save_to_disk("./local_dataset")