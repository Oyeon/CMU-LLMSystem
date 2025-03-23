# -----------------------------
# data_parallel/dataset.py
# -----------------------------
from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# ASSIGNMENT 4.1
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Given 'idx', retrieve the actual index from self.index
        and then return the corresponding sample from self.data.
        """
        data_idx = self.index[idx]
        return self.data[data_idx]


# ASSIGNMENT 4.1
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        """
        Partition 'data' into different slices according to 'sizes'.
        For example, if sizes = [0.25, 0.25, 0.25, 0.25], then the data
        is split evenly into 4 chunks.
        """
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        
        # Create a list of indices [0, 1, 2, ..., len(data)-1]
        indices = list(range(len(data)))
        # Shuffle indices
        rng.shuffle(indices)

        # Partition indices according to 'sizes'
        data_len = len(indices)
        current_index = 0
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indices[current_index:current_index + part_len])
            current_index += part_len
        
        # If floating splits don’t add up to all data points,
        # you can put leftover indices into the last partition
        if current_index < data_len:
            self.partitions[-1].extend(indices[current_index:])


    def use(self, partition):
        """
        Return a 'Partition' object that uses the subset of data
        corresponding to 'partition'.
        """
        return Partition(self.data, self.partitions[partition])


# ASSIGNMENT 4.1
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """
    Partition the training dataset for a single device in a multi-GPU data parallel setup.

    Args:
        rank (int):       The ID (rank) of the current GPU/process.
        world_size (int): Total number of GPUs (or processes).
        dataset:          The entire dataset to be split.
        batch_size (int): The total batch size (for all GPUs). Each GPU gets batch_size//world_size.
        collate_fn:       Custom function to collate a batch of data.

    Returns:
        DataLoader: A PyTorch DataLoader that provides the local dataset partition for 'rank'.
    """
    # 1. Partitioned batch size for this device
    local_batch_size = batch_size // world_size

    # 2. Partition sizes for each of the 'world_size' GPUs
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    # 3. Build a partitioner and get the local subset for this rank
    partitioner = DataPartitioner(dataset, sizes=partition_sizes)
    local_dataset = partitioner.use(rank)

    # 4. Wrap in a DataLoader
    dataloader = DataLoader(
        local_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return dataloader
