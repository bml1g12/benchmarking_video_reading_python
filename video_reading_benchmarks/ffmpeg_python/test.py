import ffmpeg
import numpy as np

class FFMPEGStream():
    def __init__(self, videopath):
        self.fn = videopath
        self.start = 0

        probe = ffmpeg.probe(videopath)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])

    def get_np_array(self, n_frames_to_read):
        out, _ = (
            ffmpeg
                .input(self.fn)
                .trim(start_frame=self.start, end_frame=n_frames_to_read)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                .run(capture_stdout=True)
        )
        video = (
            np.frombuffer(out, np.uint8)
                .reshape([-1, self.height, self.width, 3])
        )
        return video
