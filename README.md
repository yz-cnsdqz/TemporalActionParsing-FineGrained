# Local Temporal Action Parsing for Fine-grained Action Parsing

We propose a temporal local bilinear pooling method to replace max pooling in a temporal convolutional encoder-decoder network (see below), so as to capture higher-order statistics for our fine-grained tasks. Our bilinear pooling is learnable, decoupled and has a analytical solution to halve the dimensionality. For more details, please refer to our paper

    @inproceedings{zhangbilinear2018,
      title = {Local Temporal Bilinear Pooling for Fine-grained Action Parsing},
      author = {Zhang, Yan and Tang, Siyu and Muandet, Krikamol and Jarvers, Christian and Neumann, Heiko},
      booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
      month = jun,
      year = {2019},
      url = {https://arxiv.org/abs/1812.01922},
      month_numeric = {6}
    }

and a [video demo](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/470/demo-bilinear.m4v), which is better to be opened by vlc. We are still looking forward to optimizing the code.



## getting started
* The input to the network is the time sequence of frame-wise features. 
* The frontend file to run the code is __code/TCN_main.py__
* Put your features to the path of __features/{dataset name}/{feature name}/{Split_i}/{*.mat}__
* Put your dataset splits to the path of __splits/{dataset name}/{Split_i}/{train, test}.txt__, in which entries in the txt files should match the *.mat filenames.
* The tensorflow/keras models are implemented in __code/tf_models.py__, in which other pooling methods are also implemented but not used. 


## License
This project is licensed under the MIT License - see the LICENSE.md file for details.



## acknowledgement

Our implementation is based on the following framework. When use our github code, please cite their work as well.

# Temporal Convolutional Networks

This code implements the video- and sensor-based action segmentation models from [Temporal Convolutional Networks for Action Segmentation and Detection](https://arxiv.org/abs/1611.05267) by
[Colin Lea](http://colinlea.com/), [Michael Flynn](https://zo7.github.io/), Rene Vidal, Austin Reiter, Greg Hager 
arXiv 2016 (in-review). 

It was originally developed for use with the [50 Salads](http://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/), [GTEA](http://ai.stanford.edu/~alireza/GTEA/), [MERL Shopping](http://www.merl.com/demos/merl-shopping-dataset), and [JIGSAWS](http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) datasets. Recently we have also achieved high action segmentation performance on medical data, in robotics applications, and using accelerometer data from the [UCI Smartphone](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) dataset.

An abbreviated version of this work was described at the [ECCV 2016  Workshop on BNMW](http://bravenewmotion.github.io/).

Requirements: TensorFlow, Keras (1.1.2+)

Requirements (optional): 
* [Numba](http://numba.pydata.org/): This makes the metrics much faster to compute but can be removed is necessary.
* [LCTM](https://github.com/colincsl/LCTM): Our older Conditional Random Field-based models.

Tested on Python 3.5. May work on Python 2.7 but is untested.


### Contents (code folder)

* `TCN_main.py.` -- Main script for evaluation. I suggest interactively working with this in an iPython shell.
* `compare_predictions.py` -- Script to output stats on each set of predictions.
* `datasets.py` -- Adapters for processing specific datasets with a common interface.
* `metrics.py` -- Functions for computing other performance metrics. These usually take the form `score(P, Y, bg_class)` where `P` are the predictions, `Y` are the ground-truth labels, and `bg_class` is the background class.
* `tf_models.py` -- Models built with TensorFlow / Keras.
* `utils.py` -- Utilities for manipulating data.

### Data

The features used for many of the datasets we use are linked below. The video features are the output of a Spatial CNN trained using image and motion information as mentioned in the paper. To get features from the MERL dataset talk to Bharat Signh at UMD.

Each set of features should be placed in the ``features`` folder (e.g., `[TCN_directory]/features/GTEA/SpatialCNN/`). 

* [50 Salads (mid-level action granularity)](https://drive.google.com/open?id=0B2EDVAtaGbOtUTJpdWxOc0pEaEk)
* [50 Salads (eval/higher-level action granularity)](https://drive.google.com/open?id=0B2EDVAtaGbOtUUFISWNxMjFBQkk)
* [GTEA](https://drive.google.com/open?id=0B2EDVAtaGbOtZWpLZmo0dURHdU0)
* [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/): Email colincsl@gmail.com for permission. Can only be used for academic purposes. 
* [MERL Shopping](http://www.merl.com/demos/merl-shopping-dataset): Email Bharat Signh at UMD for features.

Each .mat file contains three or four types of data: 'Y' refers to the ground truth action labels for each sequence, 'X' is the per-frame probability as output from a Spatial CNN applied to each frame of video, 'A' is the 128-dim intermediate fully connected layer from the Spatial CNN applied at each frame, and if available 'S' is the sensor data (accelerometer signals in 50 Salads, robot kinematics in JIGSAWS). 

There are a set of corresponding splits for each dataset in `[TCN_directory]/splits/[dataset].` These should be easy to use with the dataset loader included here.

