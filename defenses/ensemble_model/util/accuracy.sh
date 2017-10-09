#!/bin/bash

#echo "==== inception v1 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_01.csv'
#
#echo "==== inception v2 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_02.csv'
#
#echo "==== inception v3 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_03.csv'
#
#echo "==== inception v4 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_04.csv'
#
#echo "==== inception_resnet_v2 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_05.csv'
#
#echo "==== resnet_v1_101 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_06.csv'
#
#echo "==== resnet_v1_152 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_07.csv'
#
#echo "==== resnet_v2_101 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_08.csv'
#
#echo "==== resnet_v2_152 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_09.csv'
#
#echo "==== vgg 16 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_10.csv'
#
#echo "==== vgg 19 ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_11.csv'
#
#echo "==== ENSEMBLE ===="
#python util/accuracy.py --input_file='images.csv' --output_file='result/output_0.csv'

echo "==== ENSEMBLE_TUNNED ===="
python util/accuracy.py --input_file='images.csv' --output_file='result/output.csv'

