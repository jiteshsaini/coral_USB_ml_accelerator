# Google Coral USB Accelerator performance with Raspberry Pi

**Updated to work with the latest Raspberry Pi OS (Trixie, Debian 13).**

<p align="left">
Read the article: <a href='https://helloworld.co.in/article/google-coral-usb-accelerator-performance-raspberry-pi-3b-3a-4b' target='_blank'>
   <img src='https://raw.githubusercontent.com/jiteshsaini/files/main/img/logo3.gif' height='40px'>
</a> Watch the video on Yotube:
<a href='https://youtu.be/mZLTtmPGyq0' target='_blank'>
   <img src='https://raw.githubusercontent.com/jiteshsaini/files/main/img/btn_youtube.png' height='40px'>
</a>
</p>

How much faster does a Raspberry Pi run a machine learning model with a Coral
USB Accelerator plugged in? This repository answers that with one model -
MobileNet V1, classifying images - run on the Pi's own CPU and then on the
Coral, and timed.

<p align="center">
   <img src="https://helloworld.co.in/1/sites/default/files/inline-images/coral-usb-accelerator-raspberry-pi-performance.jpeg">
</p>

## Install

Two commands on a Raspberry Pi running the 64-bit Raspberry Pi OS 12 or 13.

```bash
curl -fsSL https://raw.githubusercontent.com/jiteshsaini/coral_USB_ml_accelerator/master/setup_coral.sh -o setup_coral.sh
```

```bash
sudo bash setup_coral.sh
```

Downloading first, rather than piping into `sudo bash`, lets you read the script
before running it as root.

The code lands in `~/coral_USB_ml_accelerator`. Run the script again any time to
update: it moves your existing copy to `coral_USB_ml_accelerator.backup_<date>`
rather than overwriting it.

Plug the Coral into a **blue USB 3.0 port** on a Pi 4 or Pi 5 - on a USB 2.0
port it spends part of its time waiting on the bus.

Tested on **Raspberry Pi OS Trixie (Debian 13)** on a Raspberry Pi 4 Model B,
with a USB webcam.

## Run it

```bash
cd ~/coral_USB_ml_accelerator/exp
python3 classify.py           # on the CPU
python3 classify_coral.py     # on the Coral
```

On the Pi's desktop each opens a window showing the camera view, with what the
model sees written on it, and prints how long the camera, the model and the
window each took. They do this once a second, so the output can be read. Run
from a terminal without a desktop - over SSH, say - they work the same way
without the window. Ctrl+C, or closing the window, stops them. Either a USB
webcam or the Raspberry Pi camera module works.

On the CPU:

```
>>>  83.45  ms ( camera capture )
>>>  97.07  ms ( inference )
>>>  58.48  ms ( preview )
axolotl 0.2980392156862745
********************************
```

On the Coral:

```
>>>  107.99  ms ( camera capture )
>>>  5.06  ms ( inference )
>>>  57.05  ms ( preview )
conch 0.7647058823529411
********************************
```

The line to compare between the two is **inference**: the time the model takes
on one frame - here about 97 ms on the CPU and 5 ms on the Coral, on a Pi 4. The
last line is the model's best guess and how sure it is, from 0 to 1.

If the camera does not work, `python3 camera.py` checks it on its own, with no
model involved.

The two files are the same script, apart from three lines in
`classify_coral.py`, each marked *Coral change*. Those three lines are all it
takes to move a TensorFlow Lite model onto the Coral.

## The results

Inference time for MobileNet V1 on Raspberry Pi OS Trixie, 2026, measured with
[Model Garden](https://github.com/jiteshsaini/model_garden), which runs the same model:

| Raspberry Pi | CPU | Coral |
|---|---|---|
| Pi 4 Model B | 94 ms | 4 ms |
| Pi 3A+ | 360 ms | 10 ms |

The 3A+ has USB 2.0 only, so there the Coral spends much of its time waiting
on the bus.

The original experiment, in 2021, ran the same model on four boards - Pi 4
(4 GB and 8 GB), Pi 3B and Pi 3A+ - on Raspberry Pi OS Buster, timing the
camera, the model and the preview window together:

<p align="center">
   <img src="https://helloworld.co.in/1/sites/default/files/inline-images/raspberry-pi-coral-usb-accelerator-performance-results.jpg">
</p>

## What the script did

Worth knowing, both to understand the machine you now have and to do it by hand
if you prefer:

1. Brought the OS fully up to date, then installed numpy, Pillow, picamera2,
   Matplotlib (for the preview window) and git.
2. Installed the TensorFlow Lite runtime (`ai-edge-litert` 2.2.0) and OpenCV
   (the headless build - the preview window is drawn by Matplotlib).
3. Installed the Coral library, `libedgetpu`. Google's own build no longer
   works with current TensorFlow Lite - it loads, and then crashes as soon as a
   model is created - so this is the community rebuild from
   [feranick/libedgetpu](https://github.com/feranick/libedgetpu), which matches
   `ai-edge-litert` 2.2.0. The two versions go together.
4. Fetched this code into `~/coral_USB_ml_accelerator`.
5. Added you to the `video` group, for the camera, and to `plugdev`, which the
   Coral library's rule gives the accelerator. Nothing needs to run as root.

If the Coral was already plugged in during the install, unplug it and plug it
back in: the Coral library's permission rule only applies to devices it sees
arrive.

## Files

| File | Role |
|---|---|
| `exp/classify.py` | Classifies camera frames on the CPU |
| `exp/classify_coral.py` | The same, on the Coral |
| `exp/camera.py` | Opens a USB webcam or the camera module. Run it on its own to check the camera works |
| `exp/mobilenet_v1_1.0_224_quant.tflite` | MobileNet V1, for the CPU |
| `exp/mobilenet_v1_1.0_224_quant_edgetpu.tflite` | The same model, compiled for the Coral |
| `exp/labels_mobilenet_quant_v1_224.txt` | The 1,000 names the model can give |
| `setup_coral.sh` | The installer above |
