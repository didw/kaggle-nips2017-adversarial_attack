#!/bin/bash

python defense.py \
  --input_dir=../output \
  --output_file=result/output.csv \
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
  --checkpoint_path_vgg_19=model/vgg_19/vgg_19.ckpt

