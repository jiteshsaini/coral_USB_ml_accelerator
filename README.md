# Google Coral USB Accelerator performance with Raspberry Pi 3B, 3A+ & 4B

<p align="left">
Read the article: <a href='https://helloworld.co.in/article/google-coral-usb-accelerator-performance-raspberry-pi-3b-3a-4b' target='_blank'>
   <img src='https://github.com/jiteshsaini/files/blob/main/img/logo3.gif' height='40px'>
</a> Watch the video on Yotube: 
<a href='https://youtu.be/mZLTtmPGyq0' target='_blank'>
   <img src='https://github.com/jiteshsaini/files/blob/main/img/btn_youtube.png' height='40px'>
</a>
</p>

This is an experiment to assess the performance of 4 models of Raspberry Pi while running a Machine Learning Model (MobileNet V1) to perform image classification with and without Coral USB accelerator.

## The approach
A micro SD card with Raspbian Buster Operating System is prepared and Python scripts (Test Code) for performing image classification is placed in 'home' directory. The folder 'exp' contains the test code used in this experiment.
Now, one by one the micro SD card is inserted in the following models of Raspberry Pi:-
- Pi 4 4GB (1.5 GHz, 4GB RAM)
- Pi 4 8GB (1.5 GHz, 8GB RAM)
- Pi 3B (1.2 GHz, 1GB RAM)
- Pi 3A+ (1.4 GHz, 512MB RAM)

There 02 Python files, 02 pre-trained model files and 01 label file present in the 'exp' folder. The relation is as follows:-

- classify.py works with model file mobilenet_v1_1.0_224_quant.tflite and does not require Coral USB Accelerator to be attached.

- classify_coral.py works with mobilenet_v1_1.0_224_quant_edgetpu.tflite and requires Coral USB Accelerator to be attached.

- The label file labels_mobilenet_quant_v1_224.txt is common for both the model files.

## The Setup

The picture below shows the setup which is used in the experiment.

<p align="center">
   <img src="https://helloworld.co.in/1/sites/default/files/inline-images/coral-usb-accelerator-raspberry-pi-performance.jpeg">
</p>

## The Results

Results of the experiment are summarised in the graph below.

<p align="center">
   <img src="https://helloworld.co.in/1/sites/default/files/inline-images/raspberry-pi-coral-usb-accelerator-performance-results.jpg">
</p>





