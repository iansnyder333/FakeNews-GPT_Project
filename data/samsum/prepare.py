import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

assert torch.cuda.is_available(), "CUDA is not available, please run with GPU."
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_dataset("samsum")
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["dialogue"], truncation=True),
    batched=True,
    remove_columns=["dialogue", "summary"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["summary"], truncation=True),
    batched=True,
    remove_columns=["dialogue", "summary"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["summary"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"]
)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

tokenized_dataset.save_to_disk("data/samsum/tokenized.hf")
