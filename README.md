# TIL2023 CV Qualifiers Challenge

The TIL2023 CV Qualifiers code repository

## Setting up the environment

### Clone Repository

Run this command to clone the repository  

```shell
git clone https://github.com/til-23/til-23-cv.git
```
  
### Install requirements

To install the requirements, create a virtual environment and install the [yolov5 requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt). To use our pretrained weights, you also need to install [yolov5](https://github.com/ultralytics/yolov5).

## Model Training

### Object detection

We provide yolov5 pretrained weights for you to finetune your models as a base, but you are free to use other object detection libraries. To finetune the weights on your dataset, run the following command from the yolov5 repo.

```shell
python train.py --data coco.yaml --epochs 300 --weights 'pretrained_weights.pt' --cfg yolov5n.yaml  --batch-size 128
```

You can also refer to [this tutorial](https://docs.ultralytics.com/yolov5/train_custom_data/#before-you-start) on training a yolov5 model.

### Object Re-Identification

Refer to `src/reID`. The directory contains the following files:

* `dataset.py` - This file converts your images into a `torch.utils.data.Dataset` class. You will need to have your cropped images of your plushies and in the [LFW format](http://vis-www.cs.umass.edu/lfw/) for it to be compatible.
* `transforms.py` - This file preprocesses your images to ensure they're ingestible by the model. The most important preprocessing step is to resize the image to a standard size before they're passed into the model.
* `model.py` - This file contains the Siamese Network. This is the model you will train.
* `train.py` - This file contains the code to fit your model to the dataset.
* `test.py` - This file lets you test your model on a pair of plushie images.
* `utils.py` - This file contains misc functions that you could use.
* `model.pth` - A pretrained reID model as a baseline

## Model Inference

We have created a boilerplate code that allows you to detect plushies in a scene, and ReID a particular plushie from the detected plushies:

```shell
python3 src/inference.py
```  
