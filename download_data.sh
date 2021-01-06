#!/bin/sh
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
mkdir data
gzip -d *.gz
mv train-images-* data/training_images
mv train-labels-* data/training_labels
mv t10k-images-* data/test_images
mv t10k-labels-* data/test_labels
