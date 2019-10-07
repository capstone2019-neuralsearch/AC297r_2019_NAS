Launch AWS `p2.xlarge` GPU instance with Ubuntu-18.04 AMI, and then run:

    ./install-all.sh

(need to re-login if want to run `docker` without `sudo`)

Test installation:

    docker run --gpus all nvidia/cuda:9.0-base nvidia-smi
