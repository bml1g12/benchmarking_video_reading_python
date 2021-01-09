"""Shared functions between benchmarks"""
import time

import cv2
import timing


def tranform_tmp(output_single_cam_shape_hw, img):
    """A resizing transformation function"""
    img = cv2.resize(img,
                     (output_single_cam_shape_hw[0],
                      output_single_cam_shape_hw[1]))
    return img


def cpu_blocking_call():
    """Block the CPU for an arbitrary number of calculations worth.
    The number of iterations gives a roughly similar duration to 0.005 second sleep on my machine
    but the actual number of iterations is not important as we cannot compare directly between
    a sleep blocking call and a CPU blocking call (but we can compare between different code
    with the same blocking call)."""
    iterations = 0
    while True:
        _ = 999 * 999
        iterations += 1
        if iterations > 130000:
            break


def blocking_call(io_limited, duration=0.001):
    """Emulating a time consuming process if duration is a finite value, else return instantly.

    :param bool io_limited: If True, will use time.sleep() to emulate an I/O limited producer.
    If False will use a CPU heavy calculation to emulate a CPU limited producer.
    :param float duration: how long to block for
    """
    if duration == 0:
        return
    if io_limited:
        time.sleep(duration)
    else:
        cpu_blocking_call()


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
    print(metagroupname)
    timing_group = timing.get_timing_group(metagroupname)
    time_per_frame = timing_group.summary[groupname]["mean"] / times_calculated_over_n_frames
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
