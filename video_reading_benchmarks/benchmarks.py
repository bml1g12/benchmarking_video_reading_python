"""Benchmark functions"""
import inspect
import multiprocessing as mp
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import timing
from tqdm import tqdm
import decord

import video_reading_benchmarks
from video_reading_benchmarks.camgear.camgear import CamGear
from video_reading_benchmarks.camgear.camgear_queue import CamGear as CamGearWithQueue
from video_reading_benchmarks.imutils.custom_filevideostream import FileVideoStreamWithDownsampling
from video_reading_benchmarks.multiproc.mulitprocreader import read_video_worker, consume_frame, \
    get_video_shape
# from vidgear.gears import CamGear
from video_reading_benchmarks.shared import blocking_call, tranform_tmp
from decord import VideoLoader
import av

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

        assert output_frames_returned == config["n_frames"]
        timer.stop()
        cap.release()
        del img
        del cap
        # recreate for next repeat
        cap = cv2.VideoCapture(config["video_path"])
    del cap
    cv2.destroyAllWindows()


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
        assert frames_read == config["n_frames"]
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
        assert frames_read == config["n_frames"]
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
        assert frames_read == config["n_frames"]
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


def _prepare_shared_memory(np_arr_shape):
    mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
    np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
    shared_memory = (mp_array, np_array)
    return shared_memory


def multiproc_benchmark(config):
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "multiproc_benchmark"
    np_arr_shape = get_video_shape(config["video_path"])
    shared_memory = _prepare_shared_memory(np_arr_shape)
    proc = mp.Process(target=read_video_worker,
                      args=(config["video_path"], shared_memory, config["downsample"]))
    proc.start()
    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        for _ in tqdm(range(config["n_frames"])):
            img = consume_frame(shared_memory)

            if config["show_img"]:
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break

            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])

            frames_read += 1

        assert frames_read == config["n_frames"]
        timer.stop()
        proc.terminate()
        del shared_memory
        shared_memory = _prepare_shared_memory(np_arr_shape)
        proc = mp.Process(target=read_video_worker,
                          args=(config["video_path"],
                                shared_memory,
                                config["downsample"]))
        proc.start()
    del shared_memory
    proc.terminate()
    cv2.destroyAllWindows()


def decord_sequential_cpu_benchmark(config):
    device = "cpu"
    if device == "gpu":
        ctx = decord.gpu(0)
    else:
        ctx = decord.cpu()

    vr = decord.VideoReader(config["video_path"], ctx)
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note that decord has options " \
                                      "" \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has vr.skip_frames(N) function"
    # vr = decord.VideoReader(config["video_path"], ctx,
    #                        width=resize_width,
    #                        height=resize_height)

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with tqdm(total=config["n_frames"]) as pbar:
            while frames_read < config["n_frames"]:
                try:
                    img = vr.next()
                except StopIteration:
                    break

                img = cv2.cvtColor(img.asnumpy(), cv2.COLOR_BGR2RGB)

                if config["show_img"]:
                    cv2.imshow("img", img)
                    k = cv2.waitKey(1)
                    if ord("q") == k:
                        break

                blocking_call(config["consumer_blocking_config"]["io_limited"],
                              config["consumer_blocking_config"]["duration"])

                frames_read += 1
                pbar.update()
        assert frames_read == config["n_frames"]
        timer.stop()
        del img
        del vr
        vr = decord.VideoReader(config["video_path"], ctx)


def decord_batch_cpu_benchmark(config, buffer_size):
    device = "cpu"
    if device == "gpu":
        ctx = decord.gpu(0)
    else:
        ctx = decord.cpu()

    np_arr_shape = get_video_shape(config["video_path"])

    video_loader = decord.VideoLoader([config["video_path"]], ctx,
                                      shape=(buffer_size, *np_arr_shape),
                                      interval=1, skip=1, shuffle=0)

    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note that decord has options " \
                                      "" \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has vr.skip_frames(N) function"

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with tqdm(total=config["n_frames"]) as pbar:
            for batch in video_loader:
                if frames_read >= config["n_frames"]:
                    break

                data = batch[0].asnumpy()
                for img in data:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if config["show_img"]:
                        cv2.imshow("img", img)
                        k = cv2.waitKey(1)
                        if ord("q") == k:
                            break

                    blocking_call(config["consumer_blocking_config"]["io_limited"],
                                  config["consumer_blocking_config"]["duration"])

                    frames_read += 1
                    pbar.update()
                    if frames_read >= config["n_frames"]:
                        break

        assert frames_read == config["n_frames"]
        timer.stop()
        video_loader.reset()
        del img
        del video_loader
        video_loader = decord.VideoLoader([config["video_path"]], ctx,
                                          shape=(buffer_size, *np_arr_shape),
                                          interval=1, skip=1, shuffle=0)

def pyav_benchmark(config):

    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note that decord has options " \
                                      "" \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has vr.skip_frames(N) function"



    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with av.open(config["video_path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'  # FRAME
            for img in tqdm(container.decode(stream),
                              desc=f"Decoding",
                              unit="f"
                              ):
                img.to_ndarray(format="bgr24")

                if config["show_img"]:
                    cv2.imshow("img", img)
                    k = cv2.waitKey(1)
                    if ord("q") == k:
                        break

                blocking_call(config["consumer_blocking_config"]["io_limited"],
                              config["consumer_blocking_config"]["duration"])

                frames_read += 1
                if frames_read >= config["n_frames"]:
                    break

            timer.stop()
            assert frames_read == config["n_frames"]
            del img



if __name__ == "__main__":
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
    # baseline_benchmark(config)
    # imutils_benchmark(config, 96)
    # camgears_benchmark(config, 96)
    # camgears_with_queue_benchmark(config, 96)
    # multiproc_benchmark(config)
    # decord_sequential_cpu_benchmark(config)
    # decord_batch_cpu_benchmark(config, 96)
    pyav_benchmark(config)
