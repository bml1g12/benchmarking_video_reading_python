from pathlib import Path

import timing
import pandas as pd


import video_reading_benchmarks
from video_reading_benchmarks.benchmarks import baseline_benchmark, imutils_benchmark,\
    camgears_benchmark, camgears_with_queue_benchmark, multiproc_benchmark,\
    decord_sequential_cpu_benchmark, decord_batch_cpu_benchmark, pyav_benchmark
from video_reading_benchmarks.shared import get_timings
from video_reading_benchmarks.shared import patch_threading_excepthook

_TIME = timing.get_timing_group(__name__)

def count_frames(config):
    from imutils.video import count_frames
    print("counting frames in input")
    frame_count = count_frames(config["video_path"])
    print("frames counted: ", frame_count)
    assert (config["n_frames"] * config[
        "downsample"]) <= frame_count, "The provided video must have at least n_frames"
    return frame_count

def main():
    print("enabling threading patch")
    patch_threading_excepthook()
    print("threading patch enabled")
    config = {
        "video_path":
            str(Path(video_reading_benchmarks.__file__).parent.parent.joinpath(
                "assets/video_720x480.mkv")),
        "n_frames": 1000,
        "repeats": 3,
        "resize_shape": False,# (320, 240),
        "show_img": False,
        "downsample": 1,
        "consumer_blocking_config": {"io_limited": None,
                                     "duration": 0.005},
    }

    for is_io_limited in [False, True]:
        config["consumer_blocking_config"]["io_limited"] = is_io_limited
        assert isinstance(config["consumer_blocking_config"]["io_limited"], bool)
        print("Is IO Limited benchmark?", config["consumer_blocking_config"]["io_limited"])
        #count_frames(config)

        metagroupname = "video_reading_benchmarks.benchmarks"

        print("Starting baseline baseline_benchmark")
        baseline_benchmark(config)

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

        timings = []
        timings.append(get_timings(metagroupname, "baseline_benchmark",
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
        df.to_csv(filename)


if __name__ == "__main__":
    main()
