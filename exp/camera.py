"""
Opens whichever camera is attached and returns its frames as RGB arrays.

A USB webcam is read through OpenCV. The Raspberry Pi camera module goes
through picamera2: on current Raspberry Pi OS, /dev/video0 is the raw sensor
receiver, which OpenCV can open but never gets a picture from.

It only fetches frames; the preview window is drawn by Matplotlib, in the
classify scripts.
"""

import os
import time

# Without this OpenCV prints two warnings for every device it tries.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")

import cv2


class _Webcam:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        # The driver keeps a queue of frames, so a plain read after a pause
        # returns one taken seconds earlier. Queued frames come back at once;
        # skip them until a grab has to wait for the camera - that one is current.
        for _ in range(10):
            start = time.time()
            if not self.cap.grab():
                raise RuntimeError("the webcam stopped delivering frames")
            if time.time() - start > 0.005:
                break
        ok, frame = self.cap.retrieve()
        if not ok:
            raise RuntimeError("the webcam stopped delivering frames")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class _CameraModule:
    def __init__(self, picam):
        self.picam = picam

    def read(self):
        return self.picam.capture_array()

    def close(self):
        self.picam.stop()
        self.picam.close()


def _holder(device):
    """The program that has a device open, if it is one of ours to see."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir("/proc/%s/fd" % pid):
                if os.readlink("/proc/%s/fd/%s" % (pid, fd)) == device:
                    with open("/proc/%s/cmdline" % pid, "rb") as f:
                        cmd = f.read().replace(b"\0", b" ").decode().strip()
                    return cmd, pid
        except OSError:
            continue
    return None


def _is_usb(index):
    return "/usb" in os.path.realpath("/sys/class/video4linux/video%d" % index)


def open_camera(size=(640, 480)):
    busy = None
    # The Pi lists its own video blocks as /dev/video* too, so a webcam is not
    # always video0; only USB devices are tried. One counts once it has
    # delivered a frame. Some webcams refuse the first request or two after
    # another program lets go of them, so each gets three attempts.
    for index in filter(_is_usb, range(5)):
        for attempt in range(3):
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                ok, _ = cap.read()
                if ok:
                    print("camera: USB webcam on /dev/video%d" % index)
                    return _Webcam(cap)
            elif busy is None:
                busy = _holder("/dev/video%d" % index)
            cap.release()
            time.sleep(0.5)

    try:
        from picamera2 import Picamera2
        # libcamera lists USB webcams as well, and hands them over as 2-channel
        # YUYV frames. Only a camera module belongs on this path.
        modules = [i for i, c in enumerate(Picamera2.global_camera_info())
                   if "usb@" not in c.get("Id", "")]
        picam = Picamera2(modules[0])
    except Exception:
        if busy:
            # A program paused with Ctrl+Z still holds the camera; Ctrl+C ends it.
            raise RuntimeError("the webcam is in use by: %s\n  Stop it first - Ctrl+C in "
                               "its terminal, or: kill %s" % busy)
        raise RuntimeError("no camera found: plug in a USB webcam or connect the camera module")

    # picamera2's BGR888 arrives in R,G,B byte order - what the model wants.
    picam.configure(picam.create_video_configuration(main={"size": size, "format": "BGR888"}))
    picam.start()
    time.sleep(0.5)             # let exposure settle before the first frame
    print("camera: Raspberry Pi camera module")
    return _CameraModule(picam)


if __name__ == "__main__":
    # Run on its own, this checks the camera before any model is involved.
    cam = open_camera()
    try:
        frame = cam.read()
        print("camera: working - %dx%d frame" % (frame.shape[1], frame.shape[0]))
    finally:
        cam.close()
