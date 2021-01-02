import time

from imutils.video import FileVideoStream


class FileVideoStreamWithDownsampling(FileVideoStream):
    """Extend the imutils class to include downsampling"""
    def __init__(self, path, transform=None, queue_size=128, downsample=None):
        super().__init__(path, transform=transform, queue_size=queue_size)
        if downsample:
            print(f"Downsampling such that we only return every a frame every {downsample} frames")
            self.downsample = downsample
        else:
            self.downsample = 1


    def update(self):
        frame_number = 0
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                frame_number += 1

                if (frame_number % self.downsample) != 0:
                    continue

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue
        print("File reader closing file due to end of file or stop method.")
        self.stream.release()
