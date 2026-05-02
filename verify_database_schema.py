from datasets import load_from_disk

dataset = load_from_disk("./local_dataset/00001")

print(dataset.column_names)
print(dataset[0])