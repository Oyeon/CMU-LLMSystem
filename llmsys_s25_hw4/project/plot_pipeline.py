# plot_data_pipeline.py

import os
import json
import numpy as np
from pathlib import Path
from plot import plot  # your existing plot.py

def gather(workdir, n_epochs=2):
    times = []
    tpses = []
    for epoch in range(n_epochs):
        fpath = Path(workdir) / f"eval_results_epoch{epoch}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        # must have "training_time" and "tokens_per_sec"
        times.append(data["training_time"])
        tpses.append(data["tokens_per_sec"])

    if len(times) == 0:
        return 0,0,0,0

    arr_times = np.array(times)
    arr_tpses = np.array(tpses)

    return arr_times.mean(), arr_times.std(), arr_tpses.mean(), arr_tpses.std()

def main():
    # Suppose your logs are in:
    pipe_dir = "./workdir_pp"
    model_dir = "./workdir_mp"

    # read logs
    pipe_time_m, pipe_time_s, pipe_tps_m, pipe_tps_s = gather(pipe_dir, n_epochs=2)
    model_time_m, model_time_s, model_tps_m, model_tps_s = gather(model_dir, n_epochs=2)

    print(f"Pipeline Parallel => time {pipe_time_m:.2f}±{pipe_time_s:.2f}, tps {pipe_tps_m:.0f}±{pipe_tps_s:.0f}")
    print(f"Model Parallel => time {model_time_m:.2f}±{model_time_s:.2f}, tps {model_tps_m:.0f}±{model_tps_s:.0f}")

    # 1) Plot times
    means_time = [pipe_time_m, model_time_m]
    stds_time  = [pipe_time_s, model_time_s]
    labels_time= ["Pipeline Parallel", "Model Parallel"]
    fig_name_time = "submit_figures/pp_vs_mp_time.png"
    plot(means_time, stds_time, labels_time, fig_name_time)

    # 2) Plot throughput
    means_tps = [pipe_tps_m, model_tps_m]
    stds_tps  = [pipe_tps_s, model_tps_s]
    labels_tps= ["Pipeline Parallel", "Model Parallel"]
    fig_name_tps = "submit_figures/pp_vs_mp_tps.png"
    plot(means_tps, stds_tps, labels_tps, fig_name_tps)

if __name__=="__main__":
    main()
