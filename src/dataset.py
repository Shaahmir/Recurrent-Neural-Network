import random
from collections import deque
from tqdm import tqdm

import pandas as pd
import sentencepiece as spm
import torch
import re

from config import cfg

def clean_text(text):

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

class DatasetBuilder:

    def __init__(self):

        random.seed(cfg.RANDOM_SEED)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(
            str(cfg.TOKENIZER_DIR / "chatbot.model")
        )

        self.train_shard = 0
        self.valid_shard = 0

        self.train_samples = 0
        self.valid_samples = 0

        self.reset_shard()

    def truncate(self, ids, max_len):

        if len(ids) <= max_len:
            return ids

        return (
            [ids[0]] + ids[-(max_len - 2):] + [self.sp.eos_id()]
        )

    def encode(self, text, max_len):

        ids = self.sp.encode(
            text,
            add_bos = True,
            add_eos = True
        )
        
        return self.truncate(ids, max_len)

    def reset_shard(self):

        self.train_inputs = []
        self.train_targets = []

        self.valid_inputs = []
        self.valid_targets = []

    def save_shard(self, train = True):

        if train:

            if len(self.train_inputs) == 0:
                return 
            
            torch.save({
                "input_ids": self.train_inputs,
                "target_ids": self.train_targets,
                },

                cfg.TRAIN_DIR / f"shard_{self.train_shard:05d}.pt"
            )
            
            self.train_shard += 1

            self.train_inputs.clear()
            self.train_targets.clear()

        else:

            if len(self.valid_inputs) == 0:
                return 
            
            torch.save({
                "input_ids": self.valid_inputs,
                "target_ids": self.valid_targets,
                },

                cfg.VALID_DIR / f"shard_{self.valid_shard:05d}.pt"
            )
            
            self.valid_shard += 1

            self.valid_inputs.clear()
            self.valid_targets.clear()
    
    def process_conversation(self, conversation):
        history = deque(maxlen = cfg.MAX_HISTORY)
        samples = []

        for message in conversation:

            role = message["role"]
            text = clean_text(
                message["content"]
            )

            if not text:
                continue

            if role == "user":
                history.append(f"User: {text}")
            
            elif role == "assistant":
                if len(history) == 0:
                    continue

                context = "\n".join(history)

                input_ids = self.encode(
                    context, cfg.MAX_INPUT_LENGTH
                )

                target_ids = self.encode(
                    text, cfg.MAX_TARGET_LENGTH
                )

                samples.append(
                    (input_ids, target_ids)
                )

                history.append(f"Assistant: {text}")

        return samples

    def add_sample(self, input_ids, target_ids):

        input_ids = torch.tensor(
            input_ids,
            dtype = torch.long
        )

        target_ids = torch.tensor(
            target_ids,
            dtype = torch.long
        )

        if random.random() < cfg.VALID_SPLIT:

            self.valid_inputs.append(input_ids)
            self.valid_targets.append(target_ids)

            self.valid_samples += 1

            if len(self.valid_inputs) >= cfg.SHARD_SIZE:
                self.save_shard(train = False)
        
        else:

            self.train_inputs.append(input_ids)
            self.train_targets.append(target_ids)

            self.train_samples += 1

            if len(self.train_inputs) >= cfg.SHARD_SIZE:
                self.save_shard(train = True)

    def process_parquet(self, parquet_file):

        print(f"Processing {parquet_file.name}")
        df = pd.read_parquet(parquet_file)

        df  = df[
            (df["language"] == "English") &
            (df["redacted"] == False)
        ]

        for conversation in tqdm(df["conversation"], total = len(df)):
            
            samples = self.process_conversation(conversation)
            for input_ids, target_ids in samples:

                self.add_sample(
                    input_ids,
                    target_ids
                )

    def build(self):

        parquet_files = sorted(
            cfg.RAW_DATA.glob("*.parquet")
        )

        print(f"Found {len(parquet_files)} parquet files!")

        for parquet in parquet_files:
            self.process_parquet(parquet)

        self.save_shard(train = True)
        self.save_shard(train = False)

        print(f"Train Samples: {self.train_samples}")
        print(f"Valid Samples: {self.valid_samples}")
 
if __name__ == "__main__":

    builder = DatasetBuilder()
    builder.build()

    print(builder.encode("hello World", 30))

    df = pd.read_parquet(cfg.RAW_DATA / "train-01.parquet")
    conv = df.iloc[0]["conversation"]
    pairs = builder.process_conversation(conv)

    print(len(pairs))
    print(pairs[0][0])
    print(pairs[0][1])