import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from datasets import Dataset, load_from_disk

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print("WARNING, RUN PREPARE.py FIRST, WILL NOT WORK WITHOUT")
base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bos = "<|endoftext|>"
eos = "<|EOS|>"
pad = "<|pad|>"
special_tokens_dict = {"eos_token": eos, "bos_token": bos, "pad_token": pad}

# the new token is added to the tokenizer
num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

# the model config to which we add the special tokens
config = AutoConfig.from_pretrained(
    "gpt2",
    bos_token_id=base_tokenizer.bos_token_id,
    eos_token_id=base_tokenizer.eos_token_id,
    pad_token_id=base_tokenizer.pad_token_id,
    output_hidden_states=False,
)

# the pre-trained model is loaded with the custom configuration
base_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# the model embedding is resized
base_model.resize_token_embeddings(len(base_tokenizer))
model_headlines_path = "config/headline-gpt2"
tokenized_train_dataset = load_from_disk("data/cnn_dailymail/Htrain_dataset.hf")
tokenized_val_dataset = load_from_disk("data/cnn_dailymail/Hval_dataset.hf")
training_args = TrainingArguments(
    output_dir=model_headlines_path,  # output directory
    num_train_epochs=6,  # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=16,  # batch size for evaluation
    warmup_steps=200,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=model_headlines_path,  # directory for storing logs
    prediction_loss_only=True,
    save_steps=10000,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=base_tokenizer, mlm=False)


trainer = Trainer(
    model=base_model,  # the instantiated  Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,  # training dataset
    eval_dataset=tokenized_val_dataset,  # evaluation dataset
)
trainer.train()

trainer.save_model()
base_tokenizer.save_pretrained(model_headlines_path)

base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# special tokens are defined
bos = "<|endoftext|>"
eos = "<|EOS|>"
body = "<|body|>"
additional_special_tokens = [body]

special_tokens_dict = {
    "eos_token": eos,
    "bos_token": bos,
    "pad_token": "<pad>",
    "sep_token": body,
}
#  'additional_special_tokens':additional_special_tokens}

# the new token is added to the tokenizer
num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

# model configuration to which we add the special tokens
config = AutoConfig.from_pretrained(
    "gpt2",
    bos_token_id=base_tokenizer.bos_token_id,
    eos_token_id=base_tokenizer.eos_token_id,
    pad_token_id=base_tokenizer.pad_token_id,
    sep_token_id=base_tokenizer.sep_token_id,
    output_hidden_states=False,
)

# we load the pre-trained model with custom settings
base_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# model embeding resizing
base_model.resize_token_embeddings(len(base_tokenizer))

model_articles_path = "config/article-gpt2"

tokenized_train_dataset = load_from_disk("data/cnn_dailymail/Atrain_dataset.hf")
tokenized_val_dataset = load_from_disk("data/cnn_dailymail/Aval_dataset.hf")
training_args = TrainingArguments(
    output_dir=model_articles_path,  # output directory
    num_train_epochs=2,  # total # of training epochs
    per_device_train_batch_size=5,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=200,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=model_articles_path,  # directory for storing logs
    prediction_loss_only=True,
    save_steps=10000,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=base_tokenizer, mlm=False)

trainer = Trainer(
    model=base_model,  # the instantiated  Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,  # training dataset
    eval_dataset=tokenized_val_dataset,  # evaluation dataset
)

trainer.train()

trainer.save_model()
base_tokenizer.save_pretrained(model_articles_path)
