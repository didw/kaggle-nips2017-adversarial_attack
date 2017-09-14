#!/bin/bash

#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_01.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=1
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_02.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=2
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_03.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=3
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_04.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=4
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_05.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=5
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_06.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=6
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_07.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=7
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_08.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=8
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_09.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=9
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_10.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=10
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_11.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=11
#
#python defense.py \
#  --input_dir=../output \
#  --output_file=result/output_0.csv \
#  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
#  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
#  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
#  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
#  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
#  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
#  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
#  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
#  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
#  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
#  --test_idx=0

python defense.py \
  --input_dir=../output \
  --output_file=result/output_20.csv \
  --checkpoint_path_inception_v1=model/inception_v1/inception_v1.ckpt \
  --checkpoint_path_inception_v2=model/inception_v2/inception_v2.ckpt \
  --checkpoint_path_inception_v3=model/inception_v3/inception_v3.ckpt \
  --checkpoint_path_inception_v4=model/inception_v4/inception_v4.ckpt \
  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2/inception_resnet_v2.ckpt \
  --checkpoint_path_resnet_v1_101=model/resnet_v1_101/resnet_v1_101.ckpt \
  --checkpoint_path_resnet_v1_152=model/resnet_v1_152/resnet_v1_152.ckpt \
  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
  --checkpoint_path_vgg_16=model/vgg_16/vgg_16.ckpt \
  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt \
  --test_idx=20

