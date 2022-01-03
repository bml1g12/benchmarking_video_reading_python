"""Benchmark functions"""
import inspect
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path
import timing
from tqdm import tqdm

import av
import cv2
import decord
import numpy as np

import video_reading_benchmarks
from video_reading_benchmarks.camgear.camgear import CamGear
from video_reading_benchmarks.camgear.camgear_queue import CamGear as CamGearWithQueue
from video_reading_benchmarks.ffmpeg_python.test import FFMPEGStream
from video_reading_benchmarks.ffmpeg_python.videostream_upgraded import VideoStream as \
    FFMPEGStreamUpgraded
from video_reading_benchmarks.imutils.custom_filevideostream import FileVideoStreamWithDownsampling
from video_reading_benchmarks.multiproc.mulitprocreader import read_video_worker, consume_frame, \
    get_video_shape
from video_reading_benchmarks.shared import blocking_call, tranform_tmp

# from vidgear.gears import CamGear
# from decord import VideoLoader
print(__name__)
_TIME = timing.get_timing_group(__name__)


def max_possible_fps(config):
    """The max possible FPS if video reading was instantaneous i.e.
    just a blocking application on the consumer."""
    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=3)):
        for _ in tqdm(range(config["n_frames"])):
            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])

        timer.stop()


def baseline_benchmark(config):
    """Baseline benchmark using cv2.VideoCapture()'s .read() method."""
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
    """Benchmarking the imutils library with slight modification to allow downsampling
    and resizing.

    :param dict config:
    :param int buffer_size: max number of frames that can be cached in memory
    """
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
    """Benchmarking the vidgeats.camgears library with slight modification to allow downsampling
    and resizing and custom buffersize (default 96 in the source).

    :param dict config:
    :param int buffer_size: max number of frames that can be cached in memory
    """
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
    """Benchmarking significant modification of camgears implementation to switch collections.deque
    to a queue.Queue()

    :param dict config:
    :param int buffer_size: max number of frames that can be cached in memory
    """
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

def camgears_with_queue_official_benchmark(config):
    """Benchmarking official camgears implementation vidgear==0.1.9
    Note that next update will be switching to essentially the same as
    camgears_with_queue_benchmark() implementation
     https://github.com/abhiTronix/vidgear/pull/196/commits/3f7a6fd9efc456fbdbbb3a9394c816641701e8cf


    :param dict config:
    """
    from vidgear.gears.camgear import CamGear  #pylint: disable = redefined-outer-name, reimported, import-outside-toplevel

    cap = CamGear(source=str(config["video_path"])).start()

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
        cap = CamGear(source=str(config["video_path"])).start()
    cap.stop()
    del cap
    cv2.destroyAllWindows()


def _prepare_shared_memory(np_arr_shape):
    """Utiltiy function for multiproc_benchmark()"""
    mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
    np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
    shared_memory = (mp_array, np_array)
    return shared_memory


def multiproc_benchmark(config):
    """Benchmarking a multiprocessed (not multithreaded) video reader. Uses shared memory
    as serializing queues is very slow with multiprocesing.Queue()
    https://benjamin-lowe.medium.com/using-numpy-efficiently-between-processes-1bee17dcb01"""
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
    """Benchmarking decord library with seqeuential read"""
    device = "cpu"
    if device == "gpu":
        ctx = decord.gpu(0)
    else:
        ctx = decord.cpu()

    video_reader = decord.VideoReader(config["video_path"], ctx)
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling," \
                                      " note that decord has options " \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has " \
                                      "video_reader.skip_frames(N) function"
    # video_reader = decord.VideoReader(config["video_path"], ctx,
    #                        width=resize_width,
    #                        height=resize_height)

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with tqdm(total=config["n_frames"]) as pbar:
            while frames_read < config["n_frames"]:
                try:
                    img = video_reader.next()
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
        del video_reader
        video_reader = decord.VideoReader(config["video_path"], ctx)


def decord_batch_cpu_benchmark(config, buffer_size):
    """Benchmarking decord library with a batched implementation for reaching sequentially"""
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
    assert config["downsample"] == 1, "TODO: implement downsampling, " \
                                      "note that decord has options " \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has" \
                                      " video_reader.skip_frames(N) function"

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with tqdm(total=config["n_frames"]) as pbar:
            for batch in video_loader:
                if frames_read >= config["n_frames"]:
                    break

                data = batch[0].asnumpy()
                for img in data:
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        try:
            del img  # pylint: disable = undefined-loop-variable
        except NameError:
            pass

        del video_loader
        video_loader = decord.VideoLoader([config["video_path"]], ctx,
                                          shape=(buffer_size, *np_arr_shape),
                                          interval=1, skip=1, shuffle=0)


def pyav_benchmark(config):
    """Benchmarking pyav library for sequential viddeo reading."""
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note " \
                                      "that decord has options " \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has" \
                                      " video_reader.skip_frames(N) function"

    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        frames_read = 0
        with av.open(config["video_path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # FRAME
            for img in tqdm(container.decode(stream),
                            desc="Decoding",
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
            try:
                del img  # pylint: disable = undefined-loop-variable
            except NameError:
                pass


def ffmpeg_benchmark(config):
    """Benchmarking ffmpeg-python library for video reading,
     which is a light wrapper around ffmpeg."""
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note that " \
                                      "decord has options " \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has " \
                                      "video_reader.skip_frames(N) function"

    ffmpegstream = FFMPEGStream(config["video_path"])
    ffmpeg_raw_time_taken = []
    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        time_before = time.perf_counter()
        video = ffmpegstream.get_np_array(config["n_frames"])
        time_after = time.perf_counter()

        frames_read = 0
        for img in tqdm(video):

            if config["show_img"]:
                cv2.imshow("img", img)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break

            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])
            frames_read += 1

        timer.stop()
        assert frames_read == config["n_frames"]
        print(f"NOTE: FFMPEG actually read the file at:"
              f" {config['n_frames'] / (time_after - time_before)}"
              f" FPS but the resulting FPS is much lower as we then block n_frames worth on the "
              f"consumer")
        del ffmpegstream
        ffmpegstream = FFMPEGStream(config["video_path"])
        ffmpeg_raw_time_taken.append(time_after - time_before)

    del ffmpegstream
    return ffmpeg_raw_time_taken



def ffmpeg_upgraded_benchmark(config):
    """Benchmarking ffmpeg-python library for video reading,
     which is a light wrapper around ffmpeg."""
    assert config["resize_shape"] is False, "TODO: implement tranformation of image size for " \
                                            "decord_sequential_cpu_benchmark; note it has inbuilt" \
                                            "support for this. "
    assert config["downsample"] == 1, "TODO: implement downsampling, note that " \
                                      "decord has options " \
                                      "to sample frames every N frames" \
                                      " https://github.com/dmlc/decord#videoloader" \
                                      "Also the video reader has " \
                                      "video_reader.skip_frames(N) function"

    cap = FFMPEGStreamUpgraded(config["video_path"])
    for timer in tqdm(_TIME.measure_many(inspect.currentframe().f_code.co_name,
                                         samples=config["repeats"])):
        cap.open_stream()
        frames_read = 0
        while True:
            eof, img = cap.read()
            arr = np.frombuffer(img, np.uint8).reshape(int(cap._shape[1] * 1.5), cap._shape[0])
            bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
            if eof:
                break
            if config["show_img"]:
                cv2.imshow("img", bgr)
                k = cv2.waitKey(1)
                if ord("q") == k:
                    break
            blocking_call(config["consumer_blocking_config"]["io_limited"],
                          config["consumer_blocking_config"]["duration"])
            frames_read += 1
            if frames_read == config["n_frames"]:
                break

        timer.stop()
        assert frames_read == config["n_frames"], f"frames read was {frames_read} not " \
                                                  f"{config['n_frames']}"
        del cap
        cap = FFMPEGStreamUpgraded(config["video_path"])
    del cap


if __name__ == "__main__":
    CONFIG = {
        "video_path":
            str(Path(video_reading_benchmarks.__file__).parent.parent.joinpath(
                "assets/video_1920x1080.mkv")),
        "n_frames": 1000,
        "repeats": 3,
        "resize_shape": False,  # (320, 240),
        "show_img": False,
        "downsample": 1,
        "consumer_blocking_config": {"io_limited": False,
                                     "duration": 0},
    }
    max_possible_fps(CONFIG)
    baseline_benchmark(CONFIG)
    imutils_benchmark(CONFIG, 96)
    camgears_benchmark(CONFIG, 96)
    camgears_with_queue_benchmark(CONFIG, 96)
    multiproc_benchmark(CONFIG)
    decord_sequential_cpu_benchmark(CONFIG)
    decord_batch_cpu_benchmark(CONFIG, 96)
    pyav_benchmark(CONFIG)
    ffmpeg_benchmark(CONFIG)
    camgears_with_queue_official_benchmark(CONFIG)
    ffmpeg_benchmark(CONFIG)
    ffmpeg_upgraded_benchmark(CONFIG)
