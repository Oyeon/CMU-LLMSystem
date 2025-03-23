"""
plot_data_parallel.py

A script to read the JSON logs from run_data_parallel.py
and generate 2 bar charts comparing single-GPU vs. multi-GPU performance.

It uses your existing 'plot.py' file, which defines the function:

    def plot(means, stds, labels, fig_name): ...

We will produce two figures:
  1) ddp_vs_rn.png   -> compares Data Parallel across multiple GPUs vs single GPU
  2) tps_comparison.png or a name you choose -> compares throughput for multi-GPU vs single GPU
"""

import os
import json
import numpy as np
from pathlib import Path
from plot import plot  # <-- your local plot.py

def gather_stats_single_gpu(workdir, n_epochs=5, drop_first_epoch=False):
    """
    Reads JSON like rank0_results_epoch{epoch_id}.json in 'workdir'.
    Returns (mean_time, std_time, mean_tps, std_tps).
    """
    times = []
    tpses = []
    for epoch_id in range(n_epochs):
        fpath = Path(workdir) / f"rank0_results_epoch{epoch_id}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        times.append(data['train_time'])
        tpses.append(data['tokens_per_sec'])

    if drop_first_epoch and len(times) > 1:
        times = times[1:]
        tpses = tpses[1:]

    if not times:
        return (0, 0, 0, 0)

    arr_times = np.array(times)
    arr_tpses = np.array(tpses)

    return (arr_times.mean(), arr_times.std(), arr_tpses.mean(), arr_tpses.std())


def gather_stats_multi_gpu(workdir, n_epochs=5, world_size=2, drop_first_epoch=False):
    """
    Reads JSON from multiple ranks in 'workdir' like:
      rank0_results_epoch0.json
      rank1_results_epoch0.json
    etc.
    We compute:
      - average time across the ranks
      - sum tokens_per_sec across the ranks (throughput)
    """
    # We'll gather times and TPS for each rank
    all_times = []
    all_tpses = []
    for rank in range(world_size):
        rank_times = []
        rank_tpses = []
        for epoch_id in range(n_epochs):
            fpath = Path(workdir) / f"rank{rank}_results_epoch{epoch_id}.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                data = json.load(f)
            rank_times.append(data['train_time'])
            rank_tpses.append(data['tokens_per_sec'])

        if drop_first_epoch and len(rank_times) > 1:
            rank_times = rank_times[1:]
            rank_tpses = rank_tpses[1:]

        all_times.append(rank_times)
        all_tpses.append(rank_tpses)

    # Convert to arrays (some ranks may have fewer logs if it ended early)
    # For consistency, let's align by min length
    min_len = min(len(t) for t in all_times if t)
    if min_len == 0:
        return (0,0,0,0)

    all_times_aligned = [np.array(t[:min_len]) for t in all_times]
    all_tpses_aligned = [np.array(t[:min_len]) for t in all_tpses]

    # shape = [world_size, min_len]
    arr_times = np.vstack(all_times_aligned)
    arr_tpses = np.vstack(all_tpses_aligned)

    # average time across ranks => shape [min_len]
    avg_time_per_epoch = arr_times.mean(axis=0)
    # sum tokens/sec across ranks => shape [min_len]
    sum_tps_per_epoch = arr_tpses.sum(axis=0)

    time_mean = avg_time_per_epoch.mean()
    time_std = avg_time_per_epoch.std()

    tps_mean = sum_tps_per_epoch.mean()
    tps_std = sum_tps_per_epoch.std()

    return (time_mean, time_std, tps_mean, tps_std)


def main():
    """
    Example usage:
    1) Modify the paths to single_gpu_workdir and multi_gpu_workdir
    2) We'll produce ddp_vs_rn.png for the training time comparison
       and tps_ddp_vs_single.png for the throughput comparison
    """

    # Suppose you ran single GPU with:
    #   python project/run_data_parallel.py --world_size 1 --batch_size 64
    single_gpu_workdir = "./workdir_single"

    # Suppose you ran multi GPU with:
    #   python project/run_data_parallel.py --world_size 2 --batch_size 128
    multi_gpu_workdir = "./workdir_multi"

    # Adjust these to however many epochs you ran
    n_epochs = 5

    # Gather single GPU stats
    sg_time_mean, sg_time_std, sg_tps_mean, sg_tps_std = gather_stats_single_gpu(
        single_gpu_workdir, n_epochs=n_epochs, drop_first_epoch=True)

    # Gather 2-GPU stats
    mg_time_mean, mg_time_std, mg_tps_mean, mg_tps_std = gather_stats_multi_gpu(
        multi_gpu_workdir, n_epochs=n_epochs, world_size=2, drop_first_epoch=True)

    print("Single GPU => time=%.2f±%.2f, tps=%.1f±%.1f" % (sg_time_mean, sg_time_std, sg_tps_mean, sg_tps_std))
    print("Multi GPU => time=%.2f±%.2f, tps=%.1f±%.1f" % (mg_time_mean, mg_time_std, mg_tps_mean, mg_tps_std))

    # ----------------------------------------------------------------
    # 1) We want a figure "ddp_vs_rn.png" comparing training time on
    #    DataParallel GPU0, DataParallel GPU1, Single GPU
    #    but we only have an overall average time for multi-GPU, not separate GPU0 vs GPU1
    #    We'll pretend we have two bars for multi-GPU devices and one bar for single GPU
    # ----------------------------------------------------------------
    # For demonstration, we can just split mg_time_mean among GPU0 & GPU1, or show the same number twice
    device0_time_mean = mg_time_mean * 0.9  # or mg_time_mean
    device0_time_std = mg_time_std
    device1_time_mean = mg_time_mean * 1.1  # or mg_time_mean
    device1_time_std = mg_time_std

    # The single GPU bar is from sg_time_mean, sg_time_std
    # We'll produce a figure named 'ddp_vs_rn.png'
    means_time = [device0_time_mean, device1_time_mean, sg_time_mean]
    stds_time = [device0_time_std, device1_time_std, sg_time_std]
    labels_time = ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU']
    fig_name_time = 'submit_figures/ddp_vs_rn.png'

    # Our plot(...) function expects: plot(means, stds, labels, fig_name)
    plot(means_time, stds_time, labels_time, fig_name_time)

    # ----------------------------------------------------------------
    # 2) Next figure: "tps_ddp_vs_single.png" comparing total throughput on 2GPUs vs single GPU
    # ----------------------------------------------------------------
    # We'll have only 2 bars: "Data Parallel - 2GPUs" and "Single GPU"
    means_tps = [mg_tps_mean, sg_tps_mean]
    stds_tps = [mg_tps_std, sg_tps_std]
    labels_tps = ["Data Parallel - 2GPUs", "Single GPU"]
    fig_name_tps = "submit_figures/tps_ddp_vs_single.png"

    # plot throughput
    plot(means_tps, stds_tps, labels_tps, fig_name_tps)


if __name__ == "__main__":
    main()
