"""
Source: https://github.com/roninpawn/ffmpeg_videostream/blob/master/ffmpeg_videostream.py
"""
import ffmpeg
from warnings import warn
import re
import numpy as np
# ----> REQUIRES ffmpeg.exe AND ffprobe.exe ADDED TO CALLING DIRECTORY <----

"""
VideoStream()
  - path=  the path to your video file.

  - color=  the pixel format you'd like ffmpeg to output your video's frames as.
    : By default the YUV 4:2:0 color space is selected. It is the most prevalent encoding format,
    : which means ffmpeg probably won't have to do any conversion. Additionally, YUV420p presents 
    us 
    : with a 12-bit per-pixel bytestream, making it the smallest full color stream available. Which
    : in turn makes it the fastest to ingest. 

  - bytes_per_pixel=  the number of [bits / 8] that your pixel format (color=) requires.
    : Note that this is BYTES per pixel, not BITS.
    : RGB / BGR pixel formats tend to use 3 bytes (24 bits) for each pixel.
    : YUV420p uses 1.5 bytes (12 bits).

"""


class VideoStream:
    def __init__(self, path, color='yuv420p', bytes_per_pixel=1.5):
        self.path = path
        probe = ffmpeg.probe(self.path)
        self._inspect = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        self._shape = [self._inspect['width'], self._inspect['height']]
        self._trim = {}
        self._crop = [0, 0, *self._shape]

        self._color = color
        self._bpp = bytes_per_pixel
        self._frame_bytes = 0
        self._frame = None
        self._pipe = None
        self._EOF = False

        self._si = None
        self._si_active = False
        self._si_read_key = re.compile(rb'(\[Parsed_showinfo).*] n:')

    """
    .inspect()
      - Returns a dict() containing all data found in the 'video' stream of ffprobe.
        : When invoked as .inspect("something"), returns the value of "something" from the dict().
    """

    def inspect(self, attrib=None):
        if attrib is None: return dict(self._inspect)
        return self._inspect[attrib] if attrib in self._inspect else None

    """
    .config()
      - start_hms= / end_hms=  set new start/end time(s) enabling seeking and trimming the video. 
        : Both these values accept a string composed as [-][HH:]MM:SS[.m...] or [-]S+[.m...][
        s|ms|us]
        :     ex. "1:23:45.678" = 1 hour, 23 minutes, 45 seconds, 678 milliseconds.
        :     ex. "123.45" = 123 seconds, 450 milliseconds.
        :     ex. "1234ms" = 1234 milliseconds.
        : (see https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax )

      - crop_rect=  accepts a rectangle for cropping video output. 
        : crop_rect is a list/tuple formatted as [left, top, width, height] aka [x, y, width, 
        height]

      - output_resolution=  declares a final scaling of the video, forcing it to match this 
      resolution.
        : output_resolution is a list/tuple formatted as [width, height]

      - Considerations:
        : When crop_rect is set, it overrides the .shape() of the final output resolution. This 
        is only
        : important to note if you were to request the crop in a separate call to .config(), AFTER
        : requesting the output_resolution be changed in a previous call. For example...
        : 
        : my_videostream.config(output_resolution=(1280, 720))
        : my_videostream.config(crop_rect=(0,0,720,480))
        :
        : Here, the crop will take precedent and the final output resolution (.shape()) will be 
        720x480.

    """

    def config(self, start_hms=None, end_hms=None, crop_rect=None, output_resolution=None):
        if start_hms is not None: self._trim['ss'] = start_hms
        if end_hms is not None: self._trim['to'] = end_hms
        if crop_rect is not None:
            self._crop = crop_rect
            self._shape = [crop_rect[2] - crop_rect[0], crop_rect[3] - crop_rect[1]]
        if output_resolution is not None: self._shape = output_resolution

    """
    .open_stream()
      - showinfo=  when True invokes ffmpeg's 'showinfo' filter.
        : The 'showinfo' filter provides information about each individual frame of video as it 
        is read.
        :     (see: https://ffmpeg.org/ffmpeg-filters.html#showinfo )
        :
        : Invoking 'showinfo' reduces the maximum speed a VideoStream object can acquire raw 
        frame data.
        : In many instances the speed reduction is immeasurable due to other blocking processes. 
        But for
        : the raw acquisition of frames, 'showinfo' creates additional bytes of data that must be 
        pulled 
        : from the 'stderr' pipe -- reducing throughput.

      - loglevel=  sets ffmpeg's 'stderr' output to include/exclude data being printed to console.
        :     (see: https://ffmpeg.org/ffmpeg.html#Generic-options )
        :
        : When 'showinfo' is invoked, loglevel is overridden to "info". Its data is then parsed 
        by method
        : and will not be printed to console.

      - hide_banner=  shows/hides ffmpeg's startup banner.
        : Various 'loglevel' settings implicitly silence this banner.
        : When 'showinfo' is invoked, banner data is parsed by method and will not be printed to 
        console.

      - silence_even_test=  allows suppression of console warnings by _even_test(). (see 
      _even_test())

    """

    def open_stream(self, showinfo=False, loglevel="error", hide_banner=True,
                    silence_even_test=False):
        global_args = ['-loglevel', loglevel]
        if hide_banner: global_args.append('-hide_banner')
        self._si_active = showinfo
        pipes = {'pipe_stdout': True}

        self._EOF = False
        self._si = None
        self._even_test(silence_even_test)
        self._frame_bytes = int(self._shape[0] * self._shape[1] * self._bpp)

        stream = (
            ffmpeg
                .input(self.path, **self._trim)
                .crop(*self._crop)
                .filter('scale', *self._shape))

        if self._si_active:
            global_args[1] = "info"
            pipes['pipe_stderr'] = True
            stream = ffmpeg.filter(stream, 'showinfo')

        self._pipe = (
            ffmpeg
                .output(stream, 'pipe:', format='rawvideo', pix_fmt=self._color)
                .global_args(*global_args)
                .run_async(**pipes))

    """
    .read()
      - Returns an end-of-file boolean flag, followed by a frame's worth of raw bytes from the 
      video.
        : The bytestream data returned is in no way prepared, decoded, or shaped into an array 
        structure.
        : A simple example for converting YUV420p to BGR using numpy and OpenCV is provided:
        : 
        :     eof, frame = my_videostream.read()
        :     arr = np.frombuffer(frame, np.uint8).reshape(video.shape[1] * 1.5, video.shape[0])
        :     bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
        :
        : Note: The VideoStream class can be initialized to request BGR output from ffmpeg, 
        but it is
        : slower to acquire 24 bit RGB/BGR data than to acquire 12 bit YUV pixels and convert them.
    """

    def read(self):
        if not self._EOF:
            byte_stream = self._pipe.stdout.read(self._frame_bytes)
            if len(byte_stream):
                self._frame = byte_stream
                if self._si_active: self._read_showinfo()
            else:
                self.close()
                self._EOF = True
        return self._EOF, self._frame

    """
    .showinfo()
      - When 'showinfo' is active (see .open_stream()) returns ffmpeg's per-frame information as 
      string.
        : Called without a key=, returns the complete line of data as a string.
        : Called with key=, searches the string for a match and returns the value, or None.
        :
        :     current_frame_number = my_videostream.showinfo("n")
        :
        : (see: https://ffmpeg.org/ffmpeg-filters.html#showinfo )

    """

    def showinfo(self, key=None):
        if key is None or self._si is None: return self._si
        result = re.search(fr'(?<=\s{key}:).*?(?=\s*\w+:|\r)', self._si)
        return None if result is None else result.group()

    """
    .shape()
      - Returns the final output resolution of the video.
    """

    def shape(self):
        return self._shape

    """
    .close()
     - Closes an open VideoStream pipe.
       : The same VideoStream object may be closed and reopened with .open_stream() repeatedly.
    """

    def close(self):
        if self._pipe is not None: self._pipe.terminate()

    """
    .eof()
      - Boolean indicating whether the end of the file has been reached. 
    """

    def eof(self):
        return self._EOF

    """
    _read_showinfo() : PRIVATE METHOD
      - Collects 'showinfo' data for each frame.
        : ffmpeg's 'showinfo' filter pushes per-frame information to the 'stderr' pipe. If the 
        pipe is
        : read improperly, it will block indefinitely. This method reads any data in the stderr 
        pipe,
        : discarding all of it except showinfo's frame data.

      - Reduces raw frame ingest speed.
        : Invoking 'showinfo' reduces the maximum speed a VideoStream object can acquire raw 
        frame data.
        : In many instances the speed reduction is immeasurable due to other blocking processes. 
        But for
        : the raw acquisition of frames, 'showinfo' creates additional bytes of data that must be 
        pulled 
        : from the 'stderr' pipe -- reducing throughput.
    """

    def _read_showinfo(self):
        raw_out = bytes()
        raw_info = self._pipe.stderr.readline()
        while re.search(self._si_read_key, raw_info) is None:
            raw_info = self._pipe.stderr.readline()
        raw_out += raw_info
        self._si = raw_out.decode()

    """
    _even_test() : PRIVATE METHOD
      - Checks requested output resolution for byte-legible conformity.
        : For speed and simplicity of access, one frame's-worth of data MUST present within an
        : integer's-worth of total bytes. (the last pixel's value cannot end in the middle of a 
        byte)
        : _even_test() forces the output resolution [.shape()] to have even values of both width and
        : height, if the number of bytes-per-pixel is a non-integer. (YUV420p = 12 bits = 1.5 bytes)

        : Standard resolutions, by default, all conform to this standard. But when cropping, 
        especially 
        : procedurally, you may produce a resolution that cannot be decoded. _even_test() tries to
        : avoid that. If you are cropping or setting output sizes procedurally, call the .shape()
        : method after invoking .open_stream() to collect the final, adjusted output resolution.

      - silent=  disables the console UserWarning fired when _even_test() overrides shape().
        : As a private method, 'silent' is set True by .open_stream()'s 'silence_even_test=' option.
    """

    def _even_test(self, silent=False):
        if self._bpp != self._bpp // 1:
            w, h = self._shape
            if w % 2 > 0: w -= 1
            if h % 2 > 0: h -= 1
            out = [w, h]
            if out != self._shape:
                if not silent:
                    warn(
                        "-\n\nVideoStream: 'width/height' VALUES OF 'shape()' ARE NOT EVEN "
                        "NUMBERS & BPP IS FLOAT!\n"
                        "  Forcing value(s) to lower, even integer. This may cause errors in "
                        "operations that require "
                        f"matched resolutions.\n  Use .shape() to collect altered resolution. "
                        f"{self._shape} > {out} ",
                        stacklevel=2)
                self._shape = out


if __name__ == "__main__":
    import cv2
    cap = VideoStream("assets/video_720x480.mkv")
    cap.config(output_resolution=[720, 480])
    cap.open_stream()
    print(cap.__dict__)
    while True:
        eof, img = cap.read()
        print("The array is length", len(np.frombuffer(img, np.uint8)))
        arr = np.frombuffer(img, np.uint8).reshape(int(cap._shape[1]*1.5), cap._shape[0])

        bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
        print("Before", arr.shape, "After", bgr.shape)
        assert bgr.shape == (cap._shape[1], cap._shape[0], 3), \
            f"the shape of BGR image output is {bgr.shape}, which is not the expected shape"
        if eof:
            break
        cv2.imshow("img", bgr)
        k = cv2.waitKey(1)
        if ord("q") == k:
           break
