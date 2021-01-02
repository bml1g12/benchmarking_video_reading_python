# Benchmarking Video Reading

Tested using Python 3.7.0 on Ubuntu 20.04 LTS with Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz (4-core, 8-thread CPU)   



## How To Run 

`git lfs pull` # to get video

`pip install requirements.txt`

`pip install -e setup.py`

## Output

Timings can be found in the ./timings folder.

Timings are reported over 1000 frames as `time_for_all_frames` (seconds) +/- `stddev_for_all_frames` (seconds)  with this standard deviation calculatied over 3 repeats. `time_per_frame` is calculated as `time_for_all_frames`/1000 and the FPS is calculated as 1/`time_per_frame`.


