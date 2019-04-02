
SSD: Single Shot MultiBox Detector in TensorFlow
=======
A Tensorflow implementation of [SSD](https://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu. As a classical network framework of one-stage detectors, SSD are widely used. Our code is based on [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow). The official and original Caffe code can be found in [Caffe](https://github.com/weiliu89/caffe/tree/ssd).

DATASET
-------
#VOC_DATADET<br>
You can edit the data and path information yourself in the `tf_convert_data.py` file, then run `python tf_convert_data.py`<br>
Note the previous command generated a collection of TF-Records instead of a single file in order to ease shuffling during training.<br>


