Launch AWS `p2.xlarge` GPU instance with Ubuntu-18.04 AMI, and then run:

    ./install-all.sh

Test installation:

    docker run --gpus all nvidia/cuda:9.0-base nvidia-smi
