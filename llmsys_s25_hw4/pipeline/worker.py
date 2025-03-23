import sys
from typing import List, Tuple, Dict
from queue import Queue
from threading import Thread
from contextlib import contextmanager

import torch

InQueue = Queue
OutQueue = Queue

@contextmanager
def use_device(device: torch.device):
    """
    :func:`torch.cuda.device` for either CPU or CUDA device.
    Only does something if device is CUDA.
    """
    if device.type == "cuda":
        with torch.cuda.device(device):
            yield
    else:
        yield

class Task:
    """
    A Task object wraps a 'compute' function for the pipeline worker to run.
    """
    def __init__(self, compute):
        self._compute = compute
        # preserve grad enablement for child thread
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self):
        with torch.set_grad_enabled(self._grad_enabled):
            return self._compute()

def worker(in_queue: InQueue, out_queue: OutQueue, device: torch.device) -> None:
    """
    The worker thread:
      1. Wait for a Task from in_queue
      2. Run Task.compute()
      3. Put (success, result/exception) in out_queue
      4. If Task=None, exit
    """
    with use_device(device):
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                output = task.compute()
                out_queue.put((True, (task, output)))
            except Exception:
                exc_info = sys.exc_info()
                out_queue.put((False, exc_info))
                continue

    # Indicate "done" to watchers
    out_queue.put((False, None))

def create_workers(devices: List[torch.device]) -> Tuple[List[InQueue], List[OutQueue]]:
    """
    For each device in 'devices', spawn a worker thread that runs 'worker()'.
    Return lists of input queues & output queues, aligned with devices list.
    """
    in_queues = []
    out_queues = []

    workers_dict: Dict[torch.device, Tuple[InQueue, OutQueue]] = {}

    def normalize_device(d: torch.device) -> torch.device:
        if d.type == "cuda" and d.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        if d.type == "cpu" and d.index is not None:
            return torch.device("cpu")
        return d

    for d in devices:
        d = normalize_device(d)
        if d not in workers_dict:
            inq = Queue()
            outq = Queue()
            workers_dict[d] = (inq, outq)
            t = Thread(target=worker, args=(inq, outq, d), daemon=True)
            t.start()
        in_queues.append(workers_dict[d][0])
        out_queues.append(workers_dict[d][1])

    return in_queues, out_queues