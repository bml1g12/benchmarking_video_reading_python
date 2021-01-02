from pathlib import Path

import timing
from imutils.video import count_frames

import video_reading_benchmarks
from video_reading_benchmarks.benchmarks import baseline_benchmark, imutils_benchmark,\
    camgears_benchmark
from video_reading_benchmarks.shared import get_timings
from video_reading_benchmarks.shared import patch_threading_excepthook

_TIME = timing.get_timing_group(__name__)


def main():
    print("enabling threading patch")
    patch_threading_excepthook()
    print("threading patch enabled")
    config = {
        "video_path":
            str(Path(video_reading_benchmarks.__file__).parent.parent.joinpath("assets/video.mp4")),
        "n_frames": 1000,
        "repeats": 1,
        "resize_shape": False,# (320, 240),
        "show_img": False,
        "downsample": 2,
    }
    #buffer_size:  if a buffer is used, how many images can it store at max.

    #print("counting frames in input")
    #frame_count = count_frames(config["video_path"])
    #print("frames counted: ", frame_count)
    #assert (config["n_frames"] * config[
    #    "downsample"]) <= frame_count, "The provided video must have at least n_frames"
    metagroupname = "video_reading_benchmarks.benchmarks"

    print("Starting baseline baseline_benchmark")
    baseline_benchmark(config)

    print("Starting imutils_benchmark")
    imutils_benchmark(config, buffer_size=128)

    print("Starting camgears_benchmark")
    camgears_benchmark(config, buffer_size=3)

    timings = []
    timings.append(get_timings(metagroupname, "baseline_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "imutils_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))
    timings.append(get_timings(metagroupname, "camgears_benchmark",
                               times_calculated_over_n_frames=config["n_frames"]))

if __name__ == "__main__":
    main()
