"""Benchmark functions"""
import inspect
from pathlib import Path
from functools import partial

import cv2
import timing
from imutils.video import FileVideoStream
from video_reading_benchmarks.imutils.custom_filevideostream import FileVideoStreamWithDownsampling
from tqdm import tqdm
from video_reading_benchmarks.camgear.camgear import CamGear
from video_reading_benchmarks.camgear.camgear_queue import CamGear as CamGearWithQueue
#from vidgear.gears import CamGear
from video_reading_benchmarks.shared import blocking_call

import video_reading_benchmarks

_TIME = timing.get_timing_group(__name__)


def baseline_benchmark(config):
    cap = cv2.VideoCapture(config["video_path"])
    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        raw_frames_read = 0
        output_frames_returned = 0
        with tqdm(total=config["n_frames"]) as pbar:
            while output_frames_returned < config["n_frames"]:
                ret, img = cap.read()
                if not ret:
                    break

                raw_frames_read += 1

                if (raw_frames_read % config["downsample"]) != 0:
                    continue

                if config["resize_shape"]:
                    img = cv2.resize(img,
                                     config["resize_shape"])

                if config["show_img"]:
                    cv2.imshow("img", img)
                    k = cv2.waitKey(1)
                    if ord("q") == k:
                        break

                blocking_call(config["consumer_blocking_config"]["io_limited"],
                              config["consumer_blocking_config"]["duration"])

                output_frames_returned += 1
                pbar.update()


        timer.stop()
        cap.release()
        del img
        del cap
        # recreate for next repeat
        cap = cv2.VideoCapture()
    del cap
    cv2.destroyAllWindows()

def tranform_tmp(output_single_cam_shape_hw, img):
    """A resizing transformation function"""
    img = cv2.resize(img,
                     (output_single_cam_shape_hw[0],
                      output_single_cam_shape_hw[1]) )
    return img

def imutils_benchmark(config, buffer_size):
    if config["resize_shape"]:
        tranform_f = partial(tranform_tmp, config["resize_shape"])
    else:
        tranform_f = None
    cap = FileVideoStreamWithDownsampling(path=config["video_path"],
                                          queue_size=buffer_size,
                                          transform=tranform_f,
                                          downsample=config["downsample"]).start()

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        for _ in tqdm(range(config["n_frames"])):
            img = cap.read()

            if config["show_img"]:
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break

            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])

            frames_read += 1
        timer.stop()
        cap.stop()
        del img
        del cap.Q
        del cap
        # recreate for next repeat
        cap = FileVideoStreamWithDownsampling(path=config["video_path"],
                                              queue_size=128,
                                              transform=tranform_f,
                                              downsample=config["downsample"]).start()
    cap.stop()
    del cap.Q
    del cap
    cv2.destroyAllWindows()


def camgears_benchmark(config, buffer_size):
    if config["resize_shape"]:
        tranform_f = partial(tranform_tmp, config["resize_shape"])
    else:
        tranform_f = None

    cap = CamGear(source=str(config["video_path"]),
                  transform=tranform_f,
                  downsample=config["downsample"],
                  buffer_size=buffer_size).start()

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        for _ in tqdm(range(config["n_frames"])):
            img = cap.read()
            if img is None:
                break

            if config["show_img"]:
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break

            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])

            frames_read += 1
        timer.stop()
        cap.stop()
        del img
        del cap
        # recreate for next repeat
        cap = CamGear(source=str(config["video_path"]),
                      transform=tranform_f,
                      downsample=config["downsample"],
                      buffer_size=buffer_size).start()
    cap.stop()
    del cap
    cv2.destroyAllWindows()


def camgears_with_queue_benchmark(config, buffer_size):
    if config["resize_shape"]:
        tranform_f = partial(tranform_tmp, config["resize_shape"])
    else:
        tranform_f = None

    cap = CamGearWithQueue(source=str(config["video_path"]),
                           transform=tranform_f,
                           downsample=config["downsample"],
                           buffer_size=buffer_size).start()

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        for _ in tqdm(range(config["n_frames"])):
            img = cap.read()
            if img is None:
                break

            if config["show_img"]:
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break

            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])

            frames_read += 1
        timer.stop()
        cap.stop()
        del img
        del cap
        # recreate for next repeat
        cap = CamGearWithQueue(source=str(config["video_path"]),
                               transform=tranform_f,
                               downsample=config["downsample"],
                               buffer_size=buffer_size).start()
    cap.stop()
    del cap
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config = {
        "video_path":
            str(Path(video_reading_benchmarks.__file__).parent.parent.joinpath("assets/20200901_100748_08E4.mkv")),
        "n_frames": 900,
        "repeats": 1,
        "resize_shape": False,#(320, 240),
        "show_img": True,
        "downsample": 1,
        "consumer_blocking_config": {"io_limited": False,
                                     "duration": 0.001},
    }
    #baseline_benchmark(config)
    #imutils_benchmark(config, 128)
    #camgears_benchmark(config,96)
    camgears_with_queue_benchmark(config,96)
