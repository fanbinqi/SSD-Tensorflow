
SSD: Single Shot MultiBox Detector in TensorFlow
=======
A Tensorflow implementation of [SSD](https://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu. As a classical network framework of one-stage detectors, SSD are widely used. Our code is based on [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow). The official and original Caffe code can be found in [Caffe](https://github.com/weiliu89/caffe/tree/ssd).

DATASET
-------

You can edit the data and path information yourself in the `tf_convert_data.py` file, then run `python tf_convert_data.py`<br>
Note the previous command generated a collection of TF-Records instead of a single file in order to ease shuffling during training.<br>


Pre-trained model
-------------------------------
SSD300 trained on VOC0712[balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)

Train
---------
`python train.py` You can track your training on the tensorboard real time

In addition
-------
We implemented *Mobilenet2-SSD*, you can change framework in `nets/ssd_300_mobilenet2.py` Mobilenet-v2 is an improved version of Mobilenet, but we found that it's not a big improvement for detection. we will release the detector called `FFBNet` and [FSSD](https://github.com/fanbinqi/FSSD-Mobilenet) based on mobilenet and submit it in ICIP2019. I hope it will be a good result.

