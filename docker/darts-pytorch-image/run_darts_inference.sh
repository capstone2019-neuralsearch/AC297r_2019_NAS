#!/bin/bash

git clone https://github.com/quark0/darts.git

# Download pre-trained model
# https://github.com/quark0/darts#pretrained-models
# https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html
fileid=1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc
filename=cifar10_model.pt
wget --quiet -O $filename 'https://docs.google.com/uc?export=download&id='$fileid

python3 ./darts/cnn/test.py --auxiliary --model_path ./cifar10_model.pt
