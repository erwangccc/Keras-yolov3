# Keras-yolov3

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/Keras-yolo3](https://github.com/qqwweee/keras-yolo3])
A object detection yolov3 repo implemented by tensorflow keras. You can train and predict your own dataset using this repo.

## Contents

- [Background](https://github.com/erwangccc/Keras-yolov3#background)
- [Backbone](https://github.com/erwangccc/Keras-yolov3#backbone)
- [TO DO](https://github.com/erwangccc/Keras-yolov3#to-do)
- [Requirements](https://github.com/erwangccc/Keras-yolov3#requirements)
- [Install](https://github.com/erwangccc/Keras-yolov3#install)
- [Usage](https://github.com/erwangccc/Keras-yolov3#install)
  - [Train](https://github.com/erwangccc/Keras-yolov3#train)
  - [Predict](https://github.com/erwangccc/Keras-yolov3#predict)



## Background

Keras has been embedded to Tensorflow for a long time, and google recommends using keras.

Many problem in various areas of modern society can be classified to object detection. So i implemented Yolov3 network by tensorflow keras to used in your business.

## Backbone

- [x] Darknet53
- [x] Tiny-yolov3
- [x] Mobilenetv2-yolo 

## TO DO

- [ ] Train Mobilenetv2-yolo based on COCO dataset
- [ ] Predict video

## Requirements

* python >= 3.5

* tensorflow-gpu >= 1.12.0

* tensorboard >= 1.12.2

* PIL >= 5.3.0

* numpy

## Install

Download this repo:

```shell
$ git clone https://github.com/erwangccc/Keras-yolov3.git
$ cd kerasTest
```

## Usage

### Train

1. Generate your own annotation file and class names file.

   One row for one image;

   Row format: `image_file_path box1 box2 ... boxN`;

   Box format: `x_min,y_min,x_max,y_max,class_id` (no space).

   For VOC dataset, try `python voc_annotation.py`

   Here is an example:

   ```
   path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
   path/to/img2.jpg 120,300,250,600,2
   ...
   ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`

   The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.

   ```python
   python TrainYOLO.py
   ```

   You should modify `annotation_path` `classes_path` `anchors_path` to your custom path. Maybe you should modify some epoches number you train to get a better result.

   Set `backone` to 'yolo' or other if you load 9 anchors. Default backbone is 'yolo'.

### Predict

1. Download yolo3 weights from [darknet website](https://pjreddie.com/darknet/yolo/)
2. Convert darknet weights to keras model.
3. Run yolo detection(Only support detect image for now).

```python
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python predict.py
```

`-w` means just convert weights to keras model when you convert. 

You might need to modify the model when you start to predict. 

```
yolov3 = YOLOV3((416, 416), load_model=2)
```

According to following list.

```
['yolo', 'tiny-yolo', 'mobile']
```