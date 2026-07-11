# LAVA

LAVA (Latency Adaptive Video Analytics) is a framework that enforces strict latency bounds while maintaining the inference accuracy through fine-grained encoding parameter adaptation.

![Structure](Results/structure.png)

## Environmental Instructions:

The `LAVA_environment.yml` file defines a Conda environment named `LAVA`. This environment is built for Linux-64 systems that support GPU (CUDA).
It includes the following key components:
1. Python: 3.8.18
2. PyTorch: 2.1.2 (with CUDA 12 support via nvidia-cuda* and nvidia-cudnn-cu12 packages)
3. TorchVision: 0.16.2
4. Ultralytics YOLOv8: 8.2.52
5. OpenCV: 4.9.0.80 (via opencv-python) and base lib 4.6.0

## Emulator:

We developed a data-driven emulator. In `dataset/`, `4G.txt` is the TCP throughput trace data. 
`AD.H5` is the processed video (20 minutes) data file, which stores a series of information for each video segment under different encoding parameters (Bitrate, resolution, and frame rate). 
`AD_QP.H5` is data encoded using QP, resolution, and frame rate.
Inference models are in `inference_model/`.

Download `AD_frames.zip` and extract it to the `dataset/` path.

Under `LAVA`, run `python main.py`.

Tips: You can create data files based on your own dataset, and there are related processing functions in the `utils.py`. At the same time, it supports the construction of an online full process system.

## Key Modules:

### Throughput Predictor: 
The complete version and related variants can be found in `Throughput_Predictor.py`.

### Video Size Predictor: 
Includes network structure, training settings, and pretrained models.

### Inference Accuracy Predictor: 
Includes network structure, training settings, and pretrained models.

### Online Optimization: 
In `main.py`.

## Ablation experiment：

### About three modules: 

Select `b1` in `Throughput_Predictor.py` to test the effect of without Throughput Predictor.

Run `python wo_VP.py` to test the effect of without Video Size Predictor.

Run `python wo_AP.py` to test the effect of without Inference Accuracy Predictor.

### Fine-grained ablation study:

Throughput Predictor: There are different optional weight functions in `Throughput_Predictor.py`.

Video Size Predictor and Inference Accuracy Predictor: In the `network.py`, adjust the input of the FeatureExtractor to test the contribution of different information.

## Performance Evaluation:
Under ```Results/```, run ```python plot_results.py```.

![Latency-accuracy trade-off](Results/F_1.png)

## Video Sources:

Autonomous Driving (AD): https://www.youtube.com/watch?v=dIHYeTVklu4

Industrial Safety (IS): https://www.youtube.com/watch?v=vOG1Sm58NFc and https://www.youtube.com/watch?v=lfoTLeFooR4

Traffic Congestion (TC): https://www.youtube.com/watch?v=59c6yIYIys8

Street Facility (SF): https://drive.google.com/drive/folders/1SkogSHAO80lrMThiznVMfpUlbmsua8qR?usp=sharing

##

Note: This code is released in conjunction with a manuscript currently under peer review. It is provided to substantiate the key findings but is not the final, fully-featured version. We anticipate releasing the complete codebase upon acceptance.

