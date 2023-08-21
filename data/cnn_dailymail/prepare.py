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

filepath = "data/articles1.csv"
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

tokenized_train_dataset.save_to_disk("data/Htrain_dataset.hf")
tokenized_val_dataset.save_to_disk("data/Hval_dataset.hf")


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

num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)
df = []
for filepath in [
    "data/articles1.csv",
    "data/articles2.csv",
]:
    news_df = pd.read_csv(filepath, encoding="utf-8")
    df.append(news_df)
news_df = pd.concat(df, axis=0)


def remove_publication_headline(headline, publication):
    # publication col doesn't match exactly with newspaper in title col
    if str(publication) in str(headline):
        headline = headline.split(" - ")[0]
    return headline


def process_headlines_articles(df, title_col, content_col):
    titulo_vacio = (df[title_col].str.len() == 0) | df[title_col].isna()
    contenido_vacio = (df[content_col].str.len() == 0) | df[content_col].isna()
    df = df[~titulo_vacio & ~contenido_vacio]

    # Remove publication name from title
    df[title_col] = df.apply(
        lambda row: remove_publication_headline(row[title_col], row["publication"]),
        axis=1,
    )

    # Remove headlines with less than 8 words
    titlos_len_ge8 = df[title_col].str.split().apply(lambda x: len(x)) >= 8
    df = df[titlos_len_ge8]

    # Keep the first 100 words from the content
    df[content_col] = df[content_col].str.split(" ").apply(lambda x: " ".join(x[:100]))

    # Drop duplicates
    text_df = df.drop_duplicates(subset=[title_col, content_col])[
        [title_col, content_col]
    ]

    return text_df


temp_df = process_headlines_articles(news_df, title_col="title", content_col="content")


prepare_text = lambda x: " ".join([bos, x["title"], body, x["content"], eos])
temp_df["text"] = temp_df.apply(prepare_text, axis=1)


def tokenize_function(examples):
    return base_tokenizer(examples["text"], padding=True)


df_train_news, df_val_news = train_test_split(temp_df, train_size=0.9, random_state=77)
# we load the datasets from pandas df
train_dataset = Dataset.from_pandas(df_train_news[["text"]])
val_dataset = Dataset.from_pandas(df_val_news[["text"]])

# tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=1)

tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=1)

tokenized_train_dataset.save_to_disk("data/Atrain_dataset.hf")
tokenized_val_dataset.save_to_disk("data/Aval_dataset.hf")
