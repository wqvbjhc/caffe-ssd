# Caffe-SSD
caffe & ssd   
# License and Citation

Please cite Caffe, SSD in your publications if it helps your research:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }  
	
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
	
# Resources
mobilenetv1-ssd [Ref](https://github.com/chuanqi305/MobileNet-SSD)  
mobilenetv2-ssd [Ref](https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/ssd)  
mobilenetv2-ssdlite [Ref](https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/ssdlite)  
FocalLoss [Ref](https://github.com/chuanqi305/FocalLoss)  
DepthwiseConvolution [Ref](https://github.com/yonghenglh6/DepthwiseConvolution)  

# Contents
1. [FocalLoss](#FocalLoss)
2. [mobilenetv1-ssd](#mobilenetv1-ssd)
3. [mobilenetv2-ssd(ssdlite)](#mobilenetv2-ssd ssdlite)
4. [DepthwiseConvolution](#DepthwiseConvolution)

# FocalLoss
Caffe implementation of FAIR paper "Focal Loss for Dense Object Detection" for SSD.
```
layer {
  name: "mbox_loss"
  type: "MultiBoxFocalLoss" #change the type
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  loss_param {
    normalization: VALID
  }
  focal_loss_param { #set the alpha and gamma, default is alpha=0.25, gamma=2.0
    alpha: 0.25
    gamma: 2.0
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 21
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.5
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: NONE #do not use OHEM
  }
}
```
# mobilenetv1-ssd
A caffe implementation of MobileNet-SSD detection network, with pretrained weights on VOC0712 and mAP=0.727.

Network|mAP|Download|Download
:---:|:---:|:---:|:---:
MobileNet-SSD|72.7|[train](https://drive.google.com/open?id=0B3gersZ2cHIxVFI1Rjd5aDgwOG8)|[deploy](https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc)

### Run
1. Download source code and compile (follow the SSD README).
2. Download the pretrained deploy weights from the link above.
3. Put all the files in SSD_HOME/examples/ss/MobileNetv1-SSD
4. Run demo.py to show the detection result.
5. You can run merge_bn.py to generate a no bn model, it will be much faster.

### Train your own dataset
1. Convert your own dataset to lmdb database (follow the SSD README), and create symlinks to current directory.
```
ln -s PATH_TO_YOUR_TRAIN_LMDB trainval_lmdb
ln -s PATH_TO_YOUR_TEST_LMDB test_lmdb
```
2. Create the labelmap.prototxt file and put it into current directory.
3. Use gen_model.sh to generate your own training prototxt.
4. Download the training weights from the link above, and run train.sh, after about 30000 iterations, the loss should be 1.5 - 2.5.
5. Run test.sh to evaluate the result.
6. Run merge_bn.py to generate your own no-bn caffemodel if necessary.
```
python merge_bn.py --model example/MobileNetSSD_deploy.prototxt --weights snapshot/mobilenet_iter_xxxxxx.caffemodel
```

### About some details
There are 2 primary differences between this model and [MobileNet-SSD on tensorflow](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md):
1. ReLU6 layer is replaced by ReLU.
2. For the conv11_mbox_prior layer, the anchors is [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)] vs tensorflow's [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)].

### Reproduce the result
I trained this model from a MobileNet classifier([caffemodel](https://drive.google.com/open?id=0B3gersZ2cHIxZi13UWF0OXBsZzA) and [prototxt](https://drive.google.com/open?id=0B3gersZ2cHIxWGEzbG5nSXpNQzA)) converted from [tensorflow](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz). I first trained the model on MS-COCO and then fine-tuned on VOC0712. Without MS-COCO pretraining, it can only get mAP=0.68.

# mobilenetv2-ssd(ssdlite)
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.

### Prerequisites
Tensorflow and Caffe version [SSD](https://github.com/weiliu89/caffe) is properly installed on your computer.

### Usage
0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
cd SSD_HOME/examples/ss/Mobilenetv2-SSDLite
1. Use gen_model.py to generate the train.prototxt and deploy.prototxt (or use the default prototxt).
```
python gen_model.py -s deploy -c 91 >deploy.prototxt
```
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.
3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.
4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.
5. The original tensorflow model is trained on MSCOCO dataset, maybe you need deploy.caffemodel for VOC dataset, use coco2voc.py to get deploy_voc.caffemodel.

### Train your own dataset
1. Generate the trainval_lmdb and test_lmdb from your dataset.
2. Write a labelmap.prototxt
3. Use gen_model.py to generate some prototxt files, replace the "CLASS_NUM" with class number of your own dataset.
```
python gen_model.py -s train -c CLASS_NUM >train.prototxt
python gen_model.py -s test -c CLASS_NUM >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM >deploy.prototxt
```
4. Copy coco/solver_train.prototxt and coco/train.sh to your project and start training.

### Note
There are some differences between caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in my fork of [ssd](https://github.com/chuanqi305/ssd).

# DepthwiseConvolution
Replacing the type of mobile convolution layer with "DepthwiseConvolution" is all.   
