FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim git wget \
    python3-dev python3-pip python3-wheel python3-setuptools \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# https://github.com/pypa/pip/issues/5599
RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
    torchvision==0.2.0
RUN pip3 install --no-cache-dir xarray sklearn boto3 pytest

ARG WORKDIR='/workdir'
WORKDIR $WORKDIR

# to test the reference & official DARTS implementation
ARG DARTS_REF_DIR=$WORKDIR/darts_ref
COPY ./run_darts_inference.sh $DARTS_REF_DIR/run_darts_inference.sh
COPY ./run_darts_training.sh $DARTS_REF_DIR/run_darts_training.sh
