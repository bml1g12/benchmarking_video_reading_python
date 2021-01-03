import time
import cv2

class VideoCaptureHandler():
    """Context Manager to safely create a cv2.video capture"""

    def __init__(self, filename):
        self.cap = cv2.VideoCapture(str(filename))

    def __enter__(self):
        return self.cap

    def __exit__(self, *args):
        self.cap.release()

def video_capture_generator(filename, downsample=None):
    """Context Manager to safely create a cv2.video capture

    :param int downsample: skip every downsample frames
    """
    frame_count = 0

    if downsample is None:
        downsample = 1

    with VideoCaptureHandler(filename) as cap:
        while True:
            # grab the frame from the threaded video file stream
            (grabbed, img) = cap.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                print(
                    "The video reached the end of readable frames. Exiting reading this video: %s",
                    filename)

                break
            if img is None:
                print(
                    "The video produced a None frame. Exiting reading this video: %s",
                    filename)
                break

            frame_count += 1

            if frame_count % downsample != 0:
                continue

            yield img


def read_video_worker(video_path, shared_memory_arrays, downsample):
    """A demo of a function that is reading video from numpy arrays, then storing them in a way that
    can be accessed by other processes efficiently.

    :param str video_path: path to video file to open
    :param tuple shared_memory_arrays: Machinery for sharing information between processes, but specific
    to this camera
    :param int downsample: for every n frames read, return the dowsnample (nth) frame. 1 means
    no downsampling.
    """
    print(f"A worker process for reading video file: {video_path} has started"
          f" processing data in background.")
    mp_array, np_array = shared_memory_arrays
    for img in video_capture_generator(video_path, downsample=downsample):
        mp_array.acquire()
        np_array[:] = img

    print(f"worker process finished reading video: {video_path}")


def consume_frame(shared_memory_arrays):
    mp_array, np_array = shared_memory_arrays
    img = np_array.astype("uint8")
    while True:
        try:
            mp_array.release()
            break
        # it already unlocked, wait until its locked again which means a new frame is ready
        except ValueError:
            time.sleep(0.001)
    return img

def get_video_shape(video_path):
    with VideoCaptureHandler(video_path) as cap:
        (grabbed, img) = cap.read()
        if not grabbed:
            raise IOError(f"could not read {video_path}")
    return img.shape
