import os
import pandas as pd
import numpy as np
import tiktoken
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data/cleaned.csv")
DATA_SHARD_DIR = os.path.join(os.path.dirname(__file__), "data/data_shards")
# Initialize the tokenizer
enc = tiktoken.get_encoding(
    "gpt2"
)
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


# takes single doc and tokenizes into uint32 tokens
def tokenize(doc):
    tokens = [eot]  # begin list with eot
    # makes a list of all inputs, separated by eot token and encodes
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    # changes bytes to 16
    tokens_np_unit16 = tokens_np.astype(
        np.uint16
    )
    return tokens_np_unit16


def process_shards():
    fw = pd.read_csv(DATA_DIR)
    # create directory for shards if it doesn't already exist
    os.makedirs(DATA_SHARD_DIR, exist_ok=True)
    shard_size = int(100000)
    # determines which shard we are on
    shard_index = 0
    # place holder for token bytes (empty shard)
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    # number of tokens already on the shard
    token_count = 0
    progress_bar = None

    # applies the tokenize function to each row in fw
    for i, row in enumerate(fw["text"]):
        tokens = tokenize(str(row))
        # checks if space in current shard for new tokens
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # create progress bar
            if progress_bar is None:
                # progress bar is incremented by total tokens on shard
                progress_bar = tqdm(
                    total=shard_size, unit="tokens",
                    desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))

        # if there is no space on the shard for new tokens
        else:
            # save the current shard and start new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_SHARD_DIR, f"medquad_{split}_{shard_index:06d}"
            )
            remain = shard_size - token_count
            progress_bar.update(remain)
            all_tokens_np[token_count: token_count + remain] = tokens[:remain]
            np.save(filename, all_tokens_np)
            shard_index += 1
            # restarts the progress bar for the next shard
            progress_bar = None
            # populate next shard with tokens that could not fit on last shard
            all_tokens_np[0: len(tokens) - remain] = tokens[remain:]
            token_count = len(tokens) - remain

    # writes the remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_SHARD_DIR,
                                f"medquad_{split}_{shard_index:06d}")
        np.save(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    process_shards()
