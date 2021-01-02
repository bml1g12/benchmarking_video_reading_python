"""Shared functions between benchmarks"""

import time

import numpy as np
import timing
import sys
import threading

def patch_threading_excepthook():
    """
    This is NOT needed if using the default integrations of the Sentry library.
    Installs our exception handler into the threading modules Thread object
    Inspired by https://bugs.python.org/issue1230540
    Note that by default threads which have an unhandled exception will not pass it to root logger
    so this patch faciliates that. Unhandled exceptions thus go to __main__.log
    """
    old_init = threading.Thread.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        old_run = self.run

        def run_with_our_excepthook(*args, **kwargs):
            try:
                old_run(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:  # pylint: disable = bare-except
                sys.excepthook(*sys.exc_info())
        self.run = run_with_our_excepthook
    threading.Thread.__init__ = new_init

def get_timings(metagroupname, groupname, times_calculated_over_n_frames):
    """ Get a dictionary of the mean/std and FPS of the timing group.

    :param str metagroupname: The module path for this timing group
    :param str groupname: name of the timing group function name
    :param int times_calculated_over_n_frames: The _TIME["mean"] corresponds to this integer worth
    of frames
    :return: mean/std and FPS of the timing group as a dictionary
    :rtype: dict
    """
    # mean is the time per frame in this code
    timing_group = timing.get_timing_group(metagroupname)
    time_per_frame = timing_group.summary[groupname]["mean"]/times_calculated_over_n_frames
    stddev = f"{timing_group.summary[groupname]['stddev']:.4f}"
    fps = f"{1 / time_per_frame}"
    print(f"{groupname}: time_for_all_frames: = {timing_group.summary[groupname]['mean']} +/- "
          f"{stddev}"
          f" or FPS = {fps}")
    return {"groupname": groupname,
            "time_per_frame": f"{time_per_frame:.4f}",
            "time_for_all_frames": timing_group.summary[groupname]["mean"],
            "stddev_for_all_frames": stddev,
            "fps": fps}
