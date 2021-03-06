#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

python attack_fgsm.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v1=model/inception_v1 \
  --checkpoint_path_inception_v2=model/inception_v2 \
  --checkpoint_path_inception_v3=model/inception_v3 \
  --checkpoint_path_inception_v4=model/inception_v4 \
  --checkpoint_path_inception_resnet_v2=model/inception_resnet_v2 \
  --checkpoint_path_resnet_v1_101=model/resnet_v1_101 \
  --checkpoint_path_resnet_v1_152=model/resnet_v1_152 \
  --checkpoint_path_resnet_v2_101=model/resnet_v2_101 \
  --checkpoint_path_resnet_v2_152=model/resnet_v2_152 \
  --checkpoint_path_vgg_16=model/vgg_16 \
  --checkpoint_path_vgg_19=model/vgg_19 \
  --test_idx=20
