"""
Image classification with MobileNet V1, on the Coral USB Accelerator.

This is classify.py with three changes, each marked "Coral change". They are
all it takes to move a TensorFlow Lite model from the CPU to the Coral.
Press Ctrl+C, or close the window, to stop.
"""

import os
import time

import numpy as np
# Coral change 1 of 3: load_delegate hands the model to the accelerator.
from ai_edge_litert.interpreter import Interpreter, load_delegate

import camera

HERE = os.path.dirname(os.path.abspath(__file__))
# Coral change 2 of 3: a model compiled for the Edge TPU. The Coral cannot run
# an ordinary .tflite file.
model_path = os.path.join(HERE, "mobilenet_v1_1.0_224_quant_edgetpu.tflite")
label_path = os.path.join(HERE, "labels_mobilenet_quant_v1_224.txt")

top_k_results = 2
threshold = 0.5         # below this confidence the result is shown as ___


def centre_crop(frame, size=224):
    height, width, _ = frame.shape
    top = (height - size) // 2
    left = (width - size) // 2
    return np.ascontiguousarray(frame[top:top + size, left:left + size, :])


def time_elapsed(start_time, event):
    duration = round((time.time() - start_time) * 1000, 2)
    print(">>> ", duration, " ms (", event, ")")


with open(label_path) as f:
    labels = [line.strip() for line in f]

# Coral change 3 of 3: build the interpreter with the Edge TPU delegate.
interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#-------Window to display camera view, when there is a desktop-------
plt = None
if os.environ.get("DISPLAY"):
    import matplotlib.pyplot as plt
    plt.rcParams['toolbar'] = 'None'        # just the picture, no zoom buttons
    plt.rcParams['figure.raise_window'] = False   # don't pull the window to the front every frame
    plt.ion()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('TensorFlow Lite')
    fig.suptitle('Image Classification')
    ax = plt.gca()
    ax.set_axis_off()
    preview = None
    caption = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", va="top",
                      fontsize=18, bbox=dict(facecolor="white", edgecolor="none"))
#---------------------------------------------------------------------

cam = camera.open_camera()
try:
    while True:
        start = time.time()
        frame = cam.read()
        img = centre_crop(frame)
        time_elapsed(start, "camera capture")

        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        top_k_indices = np.argsort(predictions)[::-1][:top_k_results]
        pred_max = predictions[top_k_indices[0]] / 255.0
        lbl_max = labels[top_k_indices[0]]
        time_elapsed(start, "inference")

        if plt:
            start = time.time()
            if pred_max >= threshold:
                caption.set_text(" %s (%.1f%%) " % (lbl_max, pred_max * 100))
            else:
                caption.set_text("___")
            if preview is None:
                preview = ax.imshow(frame)
                plt.tight_layout()
            else:
                preview.set_data(frame)
            plt.pause(0.001)
            time_elapsed(start, "preview")
            if not plt.fignum_exists(fig.number):
                break

        print(lbl_max, pred_max)
        print("********************************")
        time.sleep(1)       # time to read the terminal output
except KeyboardInterrupt:
    pass
finally:
    cam.close()
