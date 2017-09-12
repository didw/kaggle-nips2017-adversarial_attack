#!/bin/bash
# Download inception v3 checkpoint for fgsm attack.
mkdir inception_v3 inception_v4 inception_resnet_v2 resnet_v2_101 resnet_v2_152 vgg_19
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
mv inception_v3_2016_08_28.tar.gz inception_v3
cd inception_v3
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
cd ..

# Download inception v4 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
mv inception_v4_2016_09_09.tar.gz inception_v4
cd inception_v4
tar -xvzf inception_v4_2016_09_09.tar.gz
rm inception_v4_2016_09_09.tar.gz
cd ..

# Download inception resnet v2 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.tar.gz inception_resnet_v2
cd inception_resnet_v2
tar -xvzf inception_resnet_v2_2016_08_30.tar.gz
rm inception_resnet_v2_2016_08_30.tar.gz
cd ..

# Download resnet v2 101 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
mv resnet_v2_101_2017_04_14.tar.gz resnet_v2_101
cd resnet_v2_101
tar -xvzf resnet_v2_101_2017_04_14.tar.gz
rm resnet_v2_101_2017_04_14.tar.gz
cd ..

# Download resnet v2 152 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
mv resnet_v2_152_2017_04_14.tar.gz resnet_v2_152
cd resnet_v2_152
tar -xvzf resnet_v2_152_2017_04_14.tar.gz
rm resnet_v2_152_2017_04_14.tar.gz
cd ..

# Download vgg19 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
mkdir inception_v1
mv inception_v1_2016_08_28.tar.gz inception_v1
cd inception_v1
tar -xvzf inception_v1_2016_08_28.tar.gz
rm inception_v1_2016_08_28.tar.gz
cd ..

# Download vgg19 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
mkdir inception_v2
mv inception_v2_2016_08_28.tar.gz inception_v2
cd inception_v2
tar -xvzf inception_v2_2016_08_28.tar.gz
rm inception_v2_2016_08_28.tar.gz
cd ..

# Download vgg19 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
mkdir resnet_v1_101
mv resnet_v1_101_2016_08_28.tar.gz resnet_v1_101
cd resnet_v1_101
tar -xvzf resnet_v1_101_2016_08_28.tar.gz
rm resnet_v1_101_2016_08_28.tar.gz
cd ..

# Download vgg19 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
mkdir resnet_v1_152
mv resnet_v1_152_2016_08_28.tar.gz resnet_v1_152
cd resnet_v1_152
tar -xvzf resnet_v1_152_2016_08_28.tar.gz
rm resnet_v1_152_2016_08_28.tar.gz
cd ..
