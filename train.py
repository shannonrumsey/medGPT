import os
import time
import torch
from torch.nn import functional as F
import tiktoken
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from models import GPT, GPTConfig
import torch.optim.lr_scheduler as lr_scheduler
import sacrebleu

# Configurations ------------
enc = tiktoken.get_encoding("gpt2")
total_batch_size = 10000
B = 25  # micro batch size
T = 100  # sequence length
# tells torch what kind of precision to use for internal computations
torch.set_float32_matmul_precision("high")
max_lr = 6e-4
min_lr = max_lr * 0.1
# must warmup for AdamW optimizer
warmup_steps = 75
max_steps = 330  # this is about 1 epoch: ~3,200,000 tokens, 10,000 batch size
# ---------------------------


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(
        npt, dtype=torch.long
    )  # convert shards so they can be inputted into layers
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_proc, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_proc = num_proc
        assert split in {"train", "val"}, f"{split} not a valid split"

        # load the shards
        DATA_SHARD_DIR = os.path.join(os.path.dirname(__file__),
                                      "data/data_shards")
        # lists all files/shards under data_root folder
        shards = os.listdir(DATA_SHARD_DIR)
        shards = [
            s for s in shards if split in s
        ]  # identifies shards with corresponding split
        shards = sorted(shards)  # read data in order
        shards = [
            os.path.join(DATA_SHARD_DIR, s) for s in shards
        ]  # creates paths to each shard instead of the folder
        self.shards = shards
        assert len(shards) > 0, f"no shards found in split {split}"
        if master_process:
            print(f"found {len(shards)} for shards in split {split}")
        self.reset()

    # resets the dataloader so that we can use both train and val splits
    def reset(self):
        # state, init at shard 0
        self.current_shard = 0
        # tokens are based on shard number
        self.tokens = load_tokens(self.shards[self.current_shard])
        # state
        self.current_pos = (
            self.B * self.T * self.process_rank
        )  # initilize location is dependent on rank/thread number

    def next_batch(self):
        B, T = self.B, self.T
        # extracts the buffer from self.tokens
        buf = self.tokens[self.current_pos:self.current_pos + B * T + 1]
        # input to transformer
        x = (buf[:-1]).view(B, T)
        # labels
        y = (buf[1:]).view(B, T)
        # updates position to start at the beginning of the next batch
        self.current_pos += (
            B * T * self.num_proc
        )  # ensures that the position advances the entire chunk
        # if at the end of the batches, reset to the beginning
        if self.current_pos + (B * T * self.num_proc + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B * T * self.process_rank
        return x, y


ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # torchrun --standalone --nproc_per_node=4
    # /Users/shannon/Documents/AK_tutorials/medGPT/train.py

    ddp_rank = int(
        os.environ["RANK"]
    )  # each process that is running (which thread on the CPU)
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # total number of processes running
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    print(ddp_world_size, ddp_rank)
    # actions with this flag only handled by rank 0
    master_process = ddp_rank == 0

    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} with DDP")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        init_process_group(backend="nccl")
    else:
        init_process_group(backend="gloo")


# revert back to single CPU training
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} but not DDP")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if device_type == "cuda":
    torch.cuda.manual_seed(1337)

# make sure batch size is divisible by B * T * ddp_world_size
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), f"{total_batch_size} is not divisible"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"num of gradient accumulation steps: {grad_accum_steps}")

# memory available: B × T × n_embd < 16 GB
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_proc=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_proc=ddp_world_size, split="val"
)

# create model
model = GPT(
    GPTConfig(vocab_size=50304)
)  # Note: Make sure this is >= the max byte found in the dataset!
model.to(device)
if ddp:
    # after back pass, DDP synchronizes and averages the gradients
    # then applies this average to all ranks
    device_ids = [ddp_local_rank] if device_type == "cuda" else None
    model = DDP(model, device_ids=device_ids)
# model.module is where ddp stored raw model containing configure_optimizers
raw_model = model.module if ddp else model

# the optimizer adjusts parameters of model based on backprop gradients
optimizer = raw_model.config_optim(
    weight_decay=0.1, lr=6e-4, device_type=device_type)

scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_steps - warmup_steps,
    eta_min=min_lr
)
LOG_PATH = os.path.join(os.path.dirname(__file__), "log.txt")
# log training loss and validation loss
with open(LOG_PATH, "w") as f:
    pass  # open file and pass to clear it

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # every 10th interation we evaulute our validation loss
    # this will tell us if how much we are overfitting
    if step % 25 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():  # no gradients involved and no backward pass
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type,
                                    dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
            val_loss_accum /= ddp_world_size
        if master_process:
            print(f"step: {step}, val loss: {val_loss_accum: .4f}\n")
            with open(LOG_PATH, "a") as f:
                f.write(f"step: {step}, val loss: {val_loss_accum: .4f}\n")

    # once in a while, generate text from the model, except step 0 (noise)
    if step > 0 and step % 25 == 0 or last_step:
        model.eval()
        num_return_sequences = 1
        max_length = 50
        tokens = enc.encode("How many people are"
                            " affected by Balance Problems?")
        input_length = len(tokens)
        tokens = torch.tensor(tokens, dtype=torch.long)
        # this is our idx in the forward function of GPT
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = (
            torch.Generator(device=device)
        )  # allows for creating and using reproducible random sequences
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # each iteration adds a position/column into x
            with torch.no_grad():  # tells pytorch no backward pass
                with torch.autocast(device_type=device_type,
                                    dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                # take the logits at the last position/column
                logits = logits[:, -1, :]
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # keeps 50 highest probabilities
                # this way we are never sampling rare tokens
                topk_probs, topk_indices = torch.topk(probs, 100, dim=-1)
                # selects token from top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                # gathers corresponding indices and creates new column
                xcol = torch.gather(topk_indices, -1, ix)
                # append new column to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)

        # print the generated text and calculate BLEU score
        hypotheses = []
        reference = [["In 2008, an estimated 14.8 percent"
                      " of American adults (33.4 million)"
                      " had a balance or dizziness problem"
                      " during the past year. See statistics"
                      " about the frequency of balance and other"
                      " sensory impairments in older adults."
                      " (Centers for Disease Control and Prevention)"]]
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            predicted_tokens = tokens[input_length:]
            decoded = enc.decode(predicted_tokens)
            hypotheses.append(decoded)
        bleu = sacrebleu.corpus_bleu(hypotheses, reference)
        print(f"Actual: {reference}, Model Attempt: {hypotheses}")
        print(f"step: {step}, bleu: {bleu.score: .4f}\n")
        if master_process:
            with open(LOG_PATH, "a") as f:
                f.write(f"step: {step}, bleu: {bleu.score: .4f}\n")

    # training loop
    model.train()
    # clear gradients for next iteration because backward adds all gradients
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # disable ddp's synchronization until very last step
        if ddp:
            # equivalent to no_sync()
            model.require_backward_grad_sync = micro_step == grad_accum_steps-1
        # forward pass which computes predicted outputs
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):  # enables mixed precision and uses bfloat when possible
            logits, loss = model(x, y)
        # find the average of an individual gradient w/o having all
        loss = loss / grad_accum_steps
        # prevents from printing only the final loss of the final micro step
        loss_accum += loss.detach()  # detaches tensor from graph
        # compute gradients of loss with respect to the model parameters
        loss.backward()
    if ddp:
        # creates average of loss_accum on all ranks
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
        # cant use dist.ReduceOp.AVG with gloo so we manually compute
        loss_accum /= ddp_world_size
    # measures the magnitude of the gradients
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )
    # determine and set the learning rate
    lr = scheduler.get_last_lr()[0]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # updates model parameters to decrease loss for next time
    optimizer.step()
    # updates learning rate
    scheduler.step()
    if device_type == "cuda":
        # wait for GPU to finish work
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    print(f"One training step took: {dt} seconds")
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step: {step}, loss: {loss_accum: .4f}\n")
        print(f"step: {step}, lr: {lr:.4e}, norm: {norm:.4f}\n")
        with open(LOG_PATH, "a") as f:
            f.write(f"step: {step}, loss: {loss_accum: .4f}\n")
            f.write(f"step: {step}, lr: {lr:.4e}, norm: {norm:.4f}\n")
if ddp:
    destroy_process_group()
