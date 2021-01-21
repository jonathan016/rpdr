# RPDR
Thesis Code - Retail Product Detection and Recognition

---

## Introduction

This repository contains codes for my masters thesis on retail product detection and recognition. As my experiments
progressed, I realized that directly researching on both tasks are too complex given the timeline and resources I had at
the time. To that extent, I reduced my experiments' scope to recognizing retail products only as provided in the 
[GroZi-120 dataset](http://grozi.calit2.net/). An example of classification experiment with interactive explanations can
be found in [this notebook](classification_demo.ipynb).

For detection models, I used [YOLOv2](https://arxiv.org/abs/1612.08242), [YOLOv3](https://arxiv.org/abs/1804.02767), and
[SSD300](https://arxiv.org/abs/1512.02325), and their backbones are used for recognition 
experiments, which are:
- **Darknet-19** from YOLOv2,
- **Darknet-53** from YOLOv3, and
- **VGG-16** from SSD300

However, due to the limitations at the time, I was not capable of importing Darknet models' ImageNet weights. This 
resulted in standard performances by Darknet models on GroZi-120 dataset. VGG-16 achieved a better result but is way too 
complex with approximately 134 million parameters for GroZi-120 dataset.
***How can I get better results with simpler model?***

The solution was adapting SSD's and Darknet's approach for recognizing images. Following SSD, I used decimation to 
transform the first two fully connected layers on VGG-16 to dilated convolution and convolution layers, respectively.
Then I replaced the last fully connected layer with 1x1 convolutional layer and global average pooling as used in
Darknet models. The resulting combination is named **VGG-16-D** and fulfills the above research question.

---

## Results

| Model | Image Resolution | Accuracy | Parameters |
| :---- | :--------------: | -------: | ---------: |
| Darknet-19 | 224x224 | 59.83% | 19,947,576 |
| Darknet-19 | 448x448 | 58.33% | 19,947,576 |
| Darknet-53 | 256x256 | 54.42% | 40,725,784 |
| Darknet-53 | 448x448 | 49.00% | 40,725,784 |
| VGG-16 | 300x300 | 66.92% | 134,752,184 |
| **VGG-16-D** | **300x300** | **67.08%** | **20,606,904** |

---

## Reproducing Experiments

### Dependencies
The ones I used, anyway:

- Ubuntu 16.04
- Python 3.5 (provided by default in Ubuntu 16.04 - just call `python` from terminal/bash)
- Python 3.7
```shell
sudo apt install -y python3.7
sudo apt-get install -y python3-pip
sudo python3.7 -m pip install --upgrade pip
```
- PyTorch and Torchvision
```shell
sudo python3.7 -m pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
- Pandas
```shell
sudo python3.7 -m pip install pandas
```
- Shapely
```shell
sudo python3.7 -m pip install shapely
```
- OpenCV (honestly, I believe I only used the latter command, but let's be safe)
```shell
sudo python3.7 -m pip install opencv-python==4.2.0.34
sudo apt-get install -y python-opencv
``` 

### Getting the Data
Run the following commands after installing all dependencies (and **after** cloning this repository):
```shell
cd ~
sudo git clone https://github.com/jonathan016/rpdr-config-results
# Stop here for recognition training and evaluation only
cd ~ && sudo mkdir grozi_zip && cd grozi_zip
sudo wget http://grozi.calit2.net/GroZi-120/videos.zip
sudo apt install -y unzip
sudo unzip videos.zip
cd video/ && sudo mv * ../ && cd .. && sudo rm videos.zip && sudo rm -rf video/ && cd ~
sudo python ./rpdr/utils/scripts/frame_extractor.py -o ~/rpdr-config-results/data/frames/ -r ~/grozi_zip/
```
The last command extracts all frames from the provided video. Note that Shelf_16.avi file seems to be corrupted, so not 
all frames are available.

### Weight Files
The weight files for all six models can be found in https://drive.google.com/drive/folders/1rsKCZx5VQvCEY3olO4ud9Q8HFEZ93WVH?usp=sharing.

### Scripts
The following table lists all training (and simultaneous evaluation) of the used models, running in background using
`nohup`. Remove the `nohup` and `&` from the command if you do not want to run the experiments in background. 

| Model | Command |
| :---: | ------- |
| Darknet-19<br/>(224x224) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v2/recognition/initial/Darknet19_224.py -m ./ &` |
| Darknet-19<br/>(448x448) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v2/recognition/finetune/Darknet19_448.py -m ./ &` |
| Darknet-53<br/>(256x256) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v3/recognition/initial/Darknet53_256.py -m ./ &` |
| Darknet-53<br/>(448x448) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v3/recognition/finetune/Darknet53_448.py -m ./ &` |
| VGG-16<br/>(300x300) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v3/recognition/initial/VGG16_300.py -m ./ &` |
| VGG-16-D<br/>(300x300) | `sudo nohup python3.7 ./rpdr/experiments/yolo/v3/recognition/initial/VGG16D_300.py -m ./ &` |

## Remarks
- This codebase would not be implemented without the provided code from [@marvis](https://github.com/marvis) and
[@sgrvinod](https://github.com/sgrvinod) from their YOLO and SSD implementations, respectively.
- The code for YOLO seems to be incorrect, perhaps due to some modifications of my own. **Use at your own risk**.
- Any help in fixing, refactoring, or anything else is greatly appreciated and wanted via issues! 
