# python2 models/depthwised_mobilenet/transferTypeToDepthwiseConvolution.py ./examples/ssd/MobileNetv2-SSD/deploy.prototxt ./examples/ssd/MobileNetv2-SSD/deploy_depthwise.prototxt
import os,sys
caffe_path = os.path.dirname(os.path.abspath(__file__))
caffe_path = os.path.join(caffe_path,'../../python')
sys.path.append(caffe_path)
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf.text_format import Merge
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_prototxt')
    parser.add_argument('target_prototxt')

    args = parser.parse_args()
    net = caffe_pb2.NetParameter()
    Merge(open(args.source_prototxt, 'r').read(), net)
    for layer in net.layer:
        if layer.type == "Convolution":
            if layer.convolution_param.group !=1:
                layer.type = "DepthwiseConvolution"
    with open(args.target_prototxt, 'w') as tf:
        tf.write(str(net))
        
