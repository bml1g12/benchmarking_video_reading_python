# Benchmarking Video Reading

Tested using Python 3.7.0 on Ubuntu 20.04 LTS with Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz (4-core, 8-thread CPU)   

Hard disk: Sequential Read - 560 MB/s CT2000MX500SSD1 Crucial MX500 2TB 3D NAND SATA 2.5 inch 7mm (with 9.5mm adapter) Internal SSD

The `baseline` benchmark is a simple cv2.VideoCapture() reader with .read() method in serial. 

In these benchmarks I compare the speed of sequentially reading a file; if one wishes to sample frames across a file, Decord maybe a good option, especially if its part of a machine learning pipeline as it has automatic conversion to PyTorch/Tensorflow arrays. 

Note: In the below I am varying whether the CONSUMER of frames is blocked via IO or CPU limitations, whereby the consumer is calling a producer video reader library which reads a video file as fast as my hard disk allows (i.e. I am not applying a blocking call inside the reader, which might emulate e.g. reading a huge video file, but instead emulating the more common use case of an application that gets blocked during processing of that video file). 

### Low Resolution input video (480x270 pixels)

![Unblocked](timings/Unblocked_video_480x270.png)

![IO Limited](timings/IOLimited_video_480x270.png)

![CPU Limited](timings/CPULimited_video_480x270.png)

### Medium Resolution input video (720x480 pixels)

![Unblocked](timings/Unblocked_video_720x480.png)

![IO Limited](timings/IOLimited_video_720x480.png)

![CPU Limited](timings/CPULimited_video_720x480.png)

### High Resolution input video (1920x1080 pixels)

![Unblocked](timings/Unblocked_video_1920x1080.png)

![IO Limited](timings/IOLimited_video_1920x1080.png)

![CPU Limited](timings/CPULimited_video_1920x1080.png)

## How To Run 

`git lfs install`

`git lfs pull` # to get video

`pip install -r requirements.txt`

`pip install -e .`

## Output

Timings can be found in the ./timings folder.

Timings are reported over 1000 frames as `time_for_all_frames` (seconds) +/- `stddev_for_all_frames` (seconds)  with this standard deviation calculatied over 3 repeats. `time_per_frame` is calculated as `time_for_all_frames`/1000 and the FPS is calculated as 1/`time_per_frame`.

# Future Work

Would like to investigate this library: https://github.com/KevinKecc/diva_io/blob/master/docs/speed.md

Would like to also investigate this library: https://abhitronix.github.io/deffcode/latest/examples/basic/

> Good news everyone🎉[...] FFMPEG based DeFFcode Video-Decoder Python Library for resolving this horrible seeking problem. Here's a example to get started:
https://abhitronix.github.io/deffcode/latest/examples/basic/#saving-keyframes-as-image to achieve precise and robust frame seeking to any time in video.
Furthermore, You can easily replace OpenCV's Videocapture API with DeFFcode APIs and enjoy all FFmpeg parameters at your fingertips. https://github.com/opencv/opencv/issues/9053#issuecomment-1075834722
