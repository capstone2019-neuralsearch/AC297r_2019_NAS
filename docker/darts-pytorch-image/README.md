Build image:

    docker build -t darts-pytorch .

Use image:

    docker run --rm -it --gpus all darts-pytorch  # just remove  `--gpus all` when using CPU
    python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    # should print (0.3.1, True)

    ./run_darts_inference.sh