import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
from tqdm import tqdm

# --------------------------------------------#
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 64  # 16 what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 1000
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 320  # 192
n_head = 6
n_layer = 6
dropout = 0.2
# --------------------------------------------#


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, channels)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # OpenAI uses 4 * n_embd
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.l1n = nn.LayerNorm(n_embd)
        self.l2n = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.l1n(x))
        x = x + self.ffwd(self.l2n(x))
        return x


class NewsGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.lang_model_head = nn.Linear(n_embd, vocab_size)  # (B,T,Vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx)  # (B,T,C)
        pos_embd = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,C)
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.ln1(x)
        logits = self.lang_model_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class NewsTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join([itos[i] for i in l])


def train_NewsGPT(text, checkpoint=None, save_path=None):
    torch.manual_seed(1337)
    with open(
        "c:context.txt",
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()
    tokenizer = NewsTokenizer(text)
    model = NewsGPT(tokenizer.vocab_size)
    m = model.to(device)
    # Train and test splits
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss = checkpoint["loss"]

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        # if ran on GPU, move to CPU
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in tqdm(range(max_iters)):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if iter > 0 and save_path:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": loss,
                    },
                    save_path.joinpath(f"checkpoint_{iter}.pt"),
                )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = tokenizer.decode(m.generate(context, max_new_tokens=1000)[0].tolist())
    if save_path:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
            },
            save_path.joinpath("NewsGPT.pt"),
        )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        },
        "NewsGPT.pt",
    )


# Pre trained GPT2 model
class FakeNewsGPT:
    def __init__(self, config: dict):
        assert "Headline" in config, "Headline config missing"
        assert "Article" in config, "Article config missing"

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.headlines = GPT2LMHeadModel.from_pretrained(config["Headline"])
        self.headlines_tokenizer = GPT2Tokenizer.from_pretrained(config["Headline"])
        self.headlines_generator = GenerationConfig.from_pretrained(
            config["Headline"], "generation_config.json"
        )
        self.articles = GPT2LMHeadModel.from_pretrained(config["Article"])
        self.articles_tokenizer = GPT2Tokenizer.from_pretrained(config["Article"])
        self.articles_generator = GenerationConfig.from_pretrained(
            config["Article"], "generation_config.json"
        )

    def generate(self, from_headline: str = None, return_text: bool = False, **kwargs):
        if not from_headline:
            input_text = self.headlines_tokenizer.bos_token
            text_ids = self.headlines_tokenizer.encode(input_text, return_tensors="pt")
            text_ids = text_ids.to(self.device)
            self.headlines = self.headlines.to(self.device)
            generated_text_samples = self.headlines.generate(
                text_ids, generation_config=self.headlines_generator
            )
            from_headline = self.headlines_tokenizer.decode(
                generated_text_samples[0], skip_special_tokens=True
            )
        headline = " ".join(
            [
                self.articles_tokenizer.bos_token,
                from_headline,
                self.articles_tokenizer.sep_token,
            ]
        )

        headline = self.articles_tokenizer.encode(headline, return_tensors="pt")
        content = self.articles.generate(
            headline, **kwargs, generation_config=self.articles_generator
        )
        text = self.articles_tokenizer.decode(
            content[0], skip_special_tokens=True
        ).replace(from_headline, "")
        if return_text:
            return f"\n {from_headline} \n\n {text}"
        else:
            print(f"\n {from_headline} \n\n {text}")


if __name__ == "__main__":
    config = {
        "Headline": "/Users/iansnyder/Desktop/Projects/Transformer/config/headline-gpt2",
        "Article": "/Users/iansnyder/Desktop/Projects/Transformer/config/article-gpt2",
    }
    model = FakeNewsGPT(config)
    model.generate(
        from_headline="Facebook Plans to Crack Down on Vaccine Misinformation"
    )
