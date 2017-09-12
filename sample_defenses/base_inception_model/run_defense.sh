#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path_inception=model/inception_v3/inception_v3.ckpt \
  --checkpoint_path_resnet_v2_101=model/resnet_v2_101/resnet_v2_101.ckpt \
  --checkpoint_path_resnet_v2_152=model/resnet_v2_152/resnet_v2_152.ckpt \
  --checkpoint_path_vgg=model/vgg_16/vgg_16.ckpt
