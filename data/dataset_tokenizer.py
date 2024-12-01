from multiprocessing import Pool, cpu_count
from datasets import load_dataset
import numpy as np
import tiktoken
import pathlib
from tqdm import tqdm

# Constants
SHARD_SIZE = int(1e8) # 100M
SHARD_DIR = pathlib.Path("data/shards")

# Tokenizer Initialization
ENCODING = tiktoken.get_encoding("gpt2")
EOT = ENCODING._special_tokens['<|endoftext|>'] # end of text token

def tokenizer(doc):
    """
    Tokenizes text and returns the tokens in an array.
    We're using uint16 to save on memory and computation
    as this is enough precision for GPT2
    """
    token_list = [EOT]
    token_list.extend(ENCODING.encode_ordinary(doc["text"]))

    tokens_array = np.array(token_list, dtype=np.uint16)
    # Create a test case for this
    # assert (tokens_array >= 0).all() and (tokens_array < 2**16).all(), "Tokens must not exceed uint16 bytes"

    return tokens_array

def save_data(token_array: np.ndarray, shard_index: int):
    """
    Save an array of tokens into a binary file to save on space
    and memoryt
    """
    shard_split = "val" if shard_index == 0 else "train"
    file = f"shard-{shard_split}-{shard_index:06d}"
    np.save(SHARD_DIR.joinpath(file), token_array)

if __name__ == '__main__':
    raw_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT")

    with Pool(max(1, cpu_count()//2)) as pool:
        shard_idx = 0
        progress_bar = None

        # Create an empty array for the tokens of a single shard
        shard_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0

        for tokens in pool.imap(tokenizer, raw_dataset['train'], chunksize=16):

            # If the shard is full of tokens, save it,
            # and move onto the next shard
            if token_count + len(tokens) >= SHARD_SIZE:
                remainder = SHARD_SIZE - token_count
                shard_tokens[token_count:token_count+remainder] = tokens[:remainder]
                save_data(shard_tokens, shard_idx)

                shard_tokens[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
                shard_idx += 1
                progress_bar = None
                continue
            
            # Fill tokens into the current non-empty shard
            shard_tokens[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_idx}")
            progress_bar.update(len(tokens))

        # Save whatever tokens remain into the last shard
        if token_count != 0:
            save_data(shard_tokens[:token_count], shard_idx)