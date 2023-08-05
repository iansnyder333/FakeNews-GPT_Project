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
from datasets import Dataset

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
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

filepath = "/content/drive/MyDrive/data/articles1.csv"
df = pd.read_csv(filepath, encoding="utf-8", usecols=["title", "publication"]).rename(
    columns={"title": "text"}
)

pd.set_option("display.max_colwidth", None)
df.head(5)


def remove_publication_headline(headline, publication):
    # publication col doesn't match exactly with newspaper in title col
    if str(publication) in str(headline):
        headline = headline.split(" - ")[0]
    return headline


def process_headlines(df, text_colname):
    # Remove empty and null rows
    titulo_vacio = (df["text"].str.len() == 0) | df["text"].isna()
    df = df[~titulo_vacio]

    # Remove publication name from title
    df["text"] = df.apply(
        lambda row: remove_publication_headline(row["text"], row["publication"]), axis=1
    )

    # Remove headlines with less than 8 words
    titlos_len_ge8 = df["text"].str.split().apply(lambda x: len(x)) >= 8
    df = df[titlos_len_ge8]

    # Drop duplicates
    text_df = df.drop_duplicates(subset=[text_colname])[[text_colname]]

    return text_df


df = process_headlines(df, "text")
df["text"] = bos + " " + df["text"] + " " + eos

df_train, df_val = train_test_split(df, train_size=0.9, random_state=77)

print(
    f"There are {len(df_train)} headlines for training and {len(df_val)} for validation"
)
# we load the datasets directly from a pandas df
train_dataset = Dataset.from_pandas(df_train[["text"]])
val_dataset = Dataset.from_pandas(df_val[["text"]])


def tokenize_function(examples):
    return base_tokenizer(examples["text"], padding=True)


tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=["text"],
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=["text"],
)
model_headlines_path = "./model_headlines_news"

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
