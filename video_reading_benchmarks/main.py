import argparse
from pathlib import Path

import timing
import time
import pandas as pd
import numpy as np

import video_reading_benchmarks
from video_reading_benchmarks.benchmarks import baseline_benchmark, imutils_benchmark, \
    camgears_benchmark, camgears_with_queue_benchmark, multiproc_benchmark, \
    decord_sequential_cpu_benchmark, decord_batch_cpu_benchmark, pyav_benchmark, ffmpeg_benchmark, \
    max_possible_fps
from video_reading_benchmarks.shared import get_timings
from video_reading_benchmarks.shared import patch_threading_excepthook

parser = argparse.ArgumentParser()
parser.add_argument("--isiolimited", action="store_true",
                    help="whether to emulate io or cpu limited consumer")

_TIME = timing.get_timing_group(__name__)


def count_frames(config):
    from imutils.video import count_frames
    print("counting frames in input")
    frame_count = count_frames(config["video_path"])
    print("frames counted: ", frame_count)
    assert (config["n_frames"] * config[
        "downsample"]) <= frame_count, "The provided video must have at least n_frames"
    return frame_count


def convert_timings_list_to_dict(groupname, timings_list, n_frames):
    timings_array = np.array(timings_list)

    return {"groupname": groupname,
            "time_per_frame": f"{timings_array.mean() / n_frames:.4f}",
            "time_for_all_frames": timings_array.mean(),
            "stddev_for_all_frames": timings_array.std(),
            "fps": (n_frames / timings_array).mean()}


def main():
    # print("enabling threading patch")
    # patch_threading_excepthook()
    # print("threading patch enabled")
    timings = []
    config = {
        "video_path":
            str(Path(video_reading_benchmarks.__file__).parent.parent.joinpath(
                "assets/video_720x480.mkv")),
        "n_frames": 1000,
        "repeats": 3,
        "resize_shape": False,  # (320, 240),
        "show_img": False,
        "downsample": 1,
        "consumer_blocking_config": {"io_limited": False,
                                     "duration": 0.005},
    }

    args = parser.parse_args()
    config["consumer_blocking_config"]["io_limited"] = args.isiolimited
    print("Is IO Limited benchmark?", config["consumer_blocking_config"]["io_limited"])
    # count_frames(config)

    metagroupname = "video_reading_benchmarks.benchmarks"

    print("Starting baseline max possible fps given the blocking consumer")
    max_possible_fps(config)

    print("Starting baseline baseline_benchmark")
    baseline_benchmark(config)

    print("Starting simple ffmpeg-python wrapper benchmark")
    ffmpeg_raw_time_taken = ffmpeg_benchmark(config)
    timings.append(convert_timings_list_to_dict("ffmpeg_unblocked_decoding_speed",
                                                ffmpeg_raw_time_taken,
                                                config["n_frames"]))

    print("pyav benchmark")
    pyav_benchmark(config)

    print("Starting multiproc_benchmark")
    multiproc_benchmark(config)

    print("Starting decord_sequential_benchmark")
    decord_sequential_cpu_benchmark(config)
    # TODO: test GPU functionality of decord

    print("Starting decord_batch_cpu_benchmark")
    decord_batch_cpu_benchmark(config, buffer_size=96)

    print("Starting imutils_benchmark")
    imutils_benchmark(config, buffer_size=96)

    print("Starting camgears_benchmark")
    camgears_benchmark(config, buffer_size=96)

    print("Starting camgears_with_queue_benchmark")
    camgears_with_queue_benchmark(config, buffer_size=96)

    timings.append(get_timings(metagroupname, "max_possible_fps",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "baseline_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "ffmpeg_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "pyav_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "multiproc_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "decord_sequential_cpu_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "decord_batch_cpu_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "imutils_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "camgears_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "camgears_with_queue_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))

    df = pd.DataFrame(timings)
    if config["consumer_blocking_config"]["io_limited"]:
        filename = "timings/benchmark_timings_iolimited.csv"
    else:
        filename = "timings/benchmark_timings_cpulimited.csv"

    df["fps"] = df["fps"].astype("float")
    df = df.sort_values("fps")
    df.to_csv(filename)
    return df


if __name__ == "__main__":
    df = main()
