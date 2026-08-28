from pathlib import Path
from torch.utils.data import Dataset
import torch
import bisect

class MultiShardDataset(Dataset):

    def __init__(self, shard_dir):

        self.shard_dir = Path(shard_dir)

        self.shards = sorted(
            self.shard_dir.glob("*.pt")
        )

        if len(self.shards) == 0:
            raise RuntimeError("No Shard Found!")

        self.index = []
        print("Builing shard index...")

        for shard in self.shards:

            data = torch.load(
                shard,
                map_location = "cpu"
            )

            count = len(data["input_ids"])

            self.index.append(
                (shard, count)
            )

            del data

        self.prefix = []
        total = 0

        for _, count in self.index:
            self.prefix.append(total)
            total += count

        self.total_samples = total

        self.current_file = None
        self.current_data = None

    def __len__(self):
        return self.total_samples
    
    def _find_shard(self, idx):

        # Binary Search
        return bisect.bisect_right(
            self.prefix,
            idx
        ) - 1

    def _load_shard(self, shard_id):

        file = self.index[shard_id][0]

        if file == self.current_file:
            return

        self.current_data = torch.load(
            file,
            map_location = "cpu"
        )

        self.current_file = file

    def __getitem__(self, index):

        shard = self._find_shard(index)
        self._load_shard(shard)

        local_idx = index - self.prefix[shard]

        x = self.current_data["input_ids"][local_idx]
        y = self.current_data["target_ids"][local_idx]

        return x, y