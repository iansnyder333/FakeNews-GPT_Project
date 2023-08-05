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
# We load the model

# options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

# We load the tokenizer
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

df = []
for filepath in [
    "/content/drive/MyDrive/data/articles1.csv",
    "/content/drive/MyDrive/data/articles2.csv",
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

model_articles_path = "./news-articles_v4"

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
