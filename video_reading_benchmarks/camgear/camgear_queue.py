# pylint: disable = superfluous-parens

"""
Modified to do image transformation on the thread, downsampling and pylinting.
Also switch to using default logger. Replaced dequeue with Queue to avoid slowdown
when CPU limited.
https://github.com/abhiTronix/vidgear/blob/master/vidgear/gears/camgear.py
by Benjamin Lowe 2021

===============================================
vidgear library source-code is deployed under the Apache 2.0 License:
Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
"""
import logging
import time
from threading import Thread

import cv2
from vidgear.gears.helper import capPropId, check_CV_version, youtube_url_validator
import queue

LOGGER = logging.getLogger(__name__)


class CamGear: #pylint: disable = too-many-instance-attributes
    """
    CamGear API supports a diverse range of video streams which can handle/control video stream
    almost any IP/USB Cameras, multimedia video file format
    (_upto 4k tested_), any network stream URL such as *http(s), rtp, rstp, rtmp, mms, etc.* In
    addition to this, it also supports live Gstreamer's RAW pipelines
    and YouTube video/livestreams URLs.
    CamGear API provides a flexible, high-level multi-threaded wrapper around [**OpenCV's
    VideoCapture Class**](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html) with
    direct access to almost all of its available parameters,
    and also internally employs `pafy` with `youtube-dl` backend for enabling seamless live
    *YouTube streaming*.
    CamGear relies exclusively on **Threaded Queue mode** for threaded, error-free and
    synchronized frame handling.
    """

    def __init__( # pylint: disable = too-many-arguments, too-many-locals, too-many-branches, too-many-statements
            self,
            source=0,
            y_tube=False,
            backend=0,
            colorspace=None,
            logging=False,
            time_delay=0,
            transform=None,
            downsample=None,
            buffer_size=96,
            **options
    ):
        if downsample is None:
            downsample = 1
        self.downsample = downsample

        self.buffer_size = buffer_size

        # enable logging if specified
        self.__logging = False
        self.transform = transform
        if logging:
            self.__logging = logging

        # check if Youtube Mode is ON (True)
        if y_tube:
            try:
                # import pafy and parse youtube stream url
                import pafy #pylint: disable = import-outside-toplevel

                # validate
                video_url = youtube_url_validator(source)
                if video_url:
                    source_object = pafy.new(video_url)
                    vo_source = source_object.getbestvideo("webm", ftypestrict=True)
                    va_source = source_object.getbest("webm", ftypestrict=False)
                    # select the best quality
                    if vo_source is None or (
                            va_source.dimensions >= vo_source.dimensions
                    ):
                        source = va_source.url
                    else:
                        source = vo_source.url
                    if self.__logging:
                        LOGGER.debug(
                            "YouTube source ID: %s, Title: %s", video_url, source_object.title
                        )
                else:
                    raise RuntimeError(
                        "Invalid `{}` Youtube URL cannot be processed!".format(source)
                    )
            except Exception as exc:
                if self.__logging:
                    LOGGER.exception(str(exc))
                raise ValueError(
                    "[CamGear:ERROR] :: YouTube Mode is enabled and the input YouTube URL is "
                    "incorrect!"
                )

        # youtube mode variable initialization
        self.__youtube_mode = y_tube

        # assigns special parameter to global variable and clear
        self.__threaded_queue_mode = options.pop("THREADED_QUEUE_MODE", True)
        if not isinstance(self.__threaded_queue_mode, bool):
            # reset improper values
            self.__threaded_queue_mode = True

        self.__queue = None
        # initialize deque for video files only
        if self.__threaded_queue_mode and isinstance(source, str):
            # import deque
            from collections import deque #pylint: disable = import-outside-toplevel

            # define deque and assign it to global var
            # max len self.buffer_size to check overflow
            self.__queue = queue.Queue(maxsize=self.buffer_size)
            # log it
            if self.__logging:
                LOGGER.debug(
                    "Enabling Threaded Queue Mode for the current video source!"
                )
        else:
            # otherwise disable it
            self.__threaded_queue_mode = False
            # log it
            if self.__logging:
                LOGGER.warning(
                    "Threaded Queue Mode is disabled for the current video source!"
                )

        # stream variable initialization
        self.stream = None

        if backend and isinstance(backend, int):
            # add backend if specified and initialize the camera stream
            if check_CV_version() == 3:
                # Different OpenCV 3.4.x statement
                self.stream = cv2.VideoCapture(source + backend)
            else:
                # Two parameters are available since OpenCV 4+ (master branch)
                self.stream = cv2.VideoCapture(source, backend)
            LOGGER.debug("Setting backend %s for this source.", backend)
        else:
            # initialize the camera stream
            self.stream = cv2.VideoCapture(source)

        # initializing colorspace variable
        self.color_space = None

        # apply attributes to source if specified
        options = {str(k).strip(): v for k, v in options.items()}
        for key, value in options.items():
            propty = capPropId(key)
            if not (propty is None):
                self.stream.set(propty, value)

        # handle colorspace value
        if not (colorspace is None):
            self.color_space = capPropId(colorspace.strip())
            if self.__logging and not (self.color_space is None):
                LOGGER.debug(
                    "Enabling %s colorspace for this video stream!",
                    colorspace.strip()
                )

        # initialize and assign frame-rate variable
        self.framerate = 0.0
        _fps = self.stream.get(cv2.CAP_PROP_FPS)
        if _fps > 1.0:
            self.framerate = _fps

        # applying time delay to warm-up webcam only if specified
        if time_delay:
            time.sleep(time_delay)

        # frame variable initialization
        (grabbed, self.frame) = self.stream.read()

        # check if valid stream
        if grabbed:
            # render colorspace if defined
            if not (self.color_space is None):
                self.frame = cv2.cvtColor(self.frame, self.color_space)

            if self.__threaded_queue_mode:

                if self.transform:
                    try:
                        self.frame = self.transform(self.frame)
                    except AttributeError:
                        LOGGER.error(
                            "[slave thread] Failed to transform the input video frame. "
                            "Setting stop state of reader.",
                            exc_info=True)
                        self.__terminate = True
                        return

                # initialize and append to queue
                self.__queue.put(self.frame)
        else:
            raise RuntimeError(
                "[CamGear:ERROR] :: Source is invalid, CamGear failed to intitialize stream on "
                "this source!"
            )

        # thread initialization
        self.__thread = None

        # initialize termination flag
        self.__terminate = False

    def start(self):
        """
        Launches the internal *Threaded Frames Extractor* daemon
        **Returns:** A reference to the CamGear class object.
        """

        self.__thread = Thread(target=self.__update, name="CamGear", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self): # pylint: disable = too-many-branches, too-many-statements
        """
        A **Threaded Frames Extractor**, that keep iterating frames from OpenCV's VideoCapture
        API to a internal monitored deque,
        until the thread is terminated, or frames runs out.
        """

        # keep iterating infinitely until the thread is terminated or frames runs out
        frame_count = 0
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.__terminate:
                break

            # otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()

            # check for valid frames
            if not grabbed:
                # no frames received, then safely exit
                if self.__threaded_queue_mode:
                    if self.__queue.qsize() == 0: #pylint: disable = no-else-break
                        print("breaking due to queue empty and failed frame grab")
                        break
                    else:
                        print("Waiting due to failed frame grab")
                        continue
                else:
                    break
            frame_count += 1
            if (frame_count % self.downsample) != 0:
                continue

            if not (self.color_space is None):
                # apply colorspace to frames
                color_frame = None
                try:
                    if isinstance(self.color_space, int):
                        color_frame = cv2.cvtColor(frame, self.color_space)
                    else:
                        if self.__logging:
                            LOGGER.warning(
                                "Global color_space parameter value %s is not a valid!",
                                self.color_space
                            )
                        self.color_space = None
                except Exception as exc: #pylint: disable = broad-except
                    # Catch if any error occurred
                    self.color_space = None
                    if self.__logging:
                        LOGGER.exception(str(exc))
                        LOGGER.warning("Input colorspace is not a valid colorspace!")
                if not (color_frame is None):
                    self.frame = color_frame
                else:
                    self.frame = frame
            else:
                self.frame = frame

            if self.transform:
                try:

                    self.frame = self.transform(frame)

                except AttributeError:

                    LOGGER.error(
                        "[slave thread] Failed to transform the input video frame. "
                        "Setting stop state of reader.",
                        exc_info=True)

                    self.__terminate = True

                    continue

            # append to queue
            if self.__threaded_queue_mode:
                self.__queue.put(self.frame)

        self.__threaded_queue_mode = False
        self.frame = None
        # release resources
        self.stream.release()

    def read(self):
        """
        Extracts frames synchronously from monitored deque, while maintaining a fixed-length
        frame buffer in the memory,
        and blocks the thread if the deque is full.
        **Returns:** A n-dimensional numpy array.
        """
        print("bml mod")
        while self.__threaded_queue_mode:
            return self.__queue.get()

        return self.frame

    def stop(self):
        """
        Safely terminates the thread, and release the VideoStream resources.
        """
        if self.__logging:
            LOGGER.debug("Terminating processes.")
        # terminate Threaded queue mode separately
        if self.__threaded_queue_mode and not (self.__queue is None):

            while not self.__queue.empty():
                try:
                    self.__queue.get(False)
                except queue.Empty:
                    continue
                self.__queue.task_done()

            self.__threaded_queue_mode = False
            self.frame = None

        # indicate that the thread should be terminate
        self.__terminate = True

        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.__thread is not None:
            self.__thread.join()
            # properly handle thread exit
            if self.__youtube_mode:
                # kill thread-lock in youtube mode
                self.__thread = None
