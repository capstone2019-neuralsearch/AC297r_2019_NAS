#!/bin/bash

git clone https://github.com/quark0/darts.git

python3 ./darts/cnn/train_search.py --unrolled  # takes very long!!
