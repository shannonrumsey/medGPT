import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # linear projection of input embeddings to 3 matrices = weight matrices
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask future positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x):
        """
        B (Batch size): Number of samples processed together in forward pass
        T (Sequence Length):  Number of tokens in each sample/sentence
        C (Embedding Dimensions): Number of features in each token
        """
        B, T, C = x.size()
        # linear transformation to get query, key, value matrices
        q, k, v = (self.c_attn(x)).split(self.n_embd, dim=2)
        # split matrices across heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Multiply the matrices together (this computes attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Reformat matrix to be compatible with next layers
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# Feed forward
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # individual layer of model (bigger than embeddings for more info)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # activation function to produce non-linearity
        self.gelu = nn.GELU(approximate="tanh")
        # maps larger hidden layer output back to orig embd dim
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer norm for attn
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # layer norm for feed forward
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = MLP(config)

    def forward(self, x):
        # adding x is to implement residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


# need to define parameters for model
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 12


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # creates n_layer number of hidden layers (blocks)
                h=nn.ModuleList([Block(config)
                                for _ in range(config.n_layer)]),
                # final normalization after final attn layer
                ln_f=nn.LayerNorm(config.n_embd)
            )
        )

        # linear transformation to get scores of vocab words
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # implement weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # apply weight initialization to all submodules
        self.apply(self._init_weights)

    # initialize MLP weights here for consistency across model
    def _init_weights(self, module):
        # override nn.Linear's default uniform init with normal
        if isinstance(module, nn.Linear):
            std = 0.02
            std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # override nn.Linear's uniform bias (optimizer updates in training)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # override nn.Embedding's default uniform init with normal
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        # B is num of sentences in sample
        # T is the sentence length
        B, T = tokens.size()
        # get all positions in sequence (in 64 bits)
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        # store positions
        pos_embd = self.transformer.wpe(pos)
        # store token embeddings
        tok_embd = self.transformer.wte(tokens)
        x = tok_embd + pos_embd
        # forward embeddings through blocks
        for block in self.transformer.h:
            x = block(x)
        # final normalization after final attn
        x = self.transformer.ln_f(x)
        # calculate predictions
        logits = self.lm_head(x)
        # calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    # optimizer to update parameters during training
    def config_optim(self, weight_decay, lr, device_type):
        # only interested in params requiring grad
        param_dict = {pn: p
                      for pn, p in self.named_parameters()
                      if p.requires_grad}
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        # these are biases, layer norm, etc.
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        param_groups = [{"params": decay_params, "weight decay": weight_decay},
                        {"params": nodecay_params, "weight decay": 0.0}]
        avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused = True if avail and device_type == "cuda" else None
        optimizer = torch.optim.AdamW(param_groups,
                                      lr=lr,
                                      betas=(0.9, 0.95),
                                      eps=1e-08,
                                      fused=fused)
        return optimizer
