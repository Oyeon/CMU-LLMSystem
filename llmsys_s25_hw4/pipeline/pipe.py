from typing import Any, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import nn
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    """
    Generate the schedule for each clock cycle.
    For clock k from 0..(num_batches+num_partitions-2),
    we collect all (i,j) s.t. i+j=k, 0<=i<num_batches, 0<=j<num_partitions.
    """
    for k in range(num_batches + num_partitions - 1):
        step = []
        for j in range(num_partitions):
            i = k - j
            if 0 <= i < num_batches:
                step.append((i, j))
        yield step


class Pipe(nn.Module):
    def __init__(self, module: nn.ModuleList, split_size: int = 1) -> None:
        super().__init__()
        self.split_size = int(split_size)
        # Split the big nn.Sequential into sub-sequences (partitions)
        self.partitions, self.devices = _split_module(module)
        # Create worker threads
        self.in_queues, self.out_queues = create_workers(self.devices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Split x into micro-batches
        micro_batches = torch.chunk(x, self.split_size, dim=0)
        batches = list(micro_batches)

        # 2) Build the schedule
        schedule = _clock_cycles(num_batches=len(batches), num_partitions=len(self.partitions))

        # 3) For each clock cycle, compute
        for clock_step in schedule:
            self.compute(batches, clock_step)

        # 4) Move final microbatches to last device & concatenate
        last_dev = self.devices[-1]
        outputs = [b.to(last_dev) for b in batches]
        return torch.cat(outputs, dim=0)

    def compute(self, batches: List[torch.Tensor], schedule: List[Tuple[int, int]]) -> None:
        tasks = []
        for (micro_idx, part_idx) in schedule:
            def compute_func(mi=micro_idx, pj=part_idx):
                # Move the microbatch to partition pj's device before calling partitions[pj]
                dev = self.devices[pj]
                x = batches[mi].to(dev)
                return self.partitions[pj](x)

            task = Task(compute_func)
            self.in_queues[part_idx].put(task)
            tasks.append((micro_idx, part_idx, task))

        # Retrieve results
        for (micro_idx, part_idx, task) in tasks:
            success, payload = self.out_queues[part_idx].get()
            if not success:
                _, exc_value, tb = payload
                raise exc_value.with_traceback(tb)
            else:
                _, batch_output = payload
                batches[micro_idx] = batch_output
