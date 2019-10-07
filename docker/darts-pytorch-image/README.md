Build image:

    docker build -t darts-pytorch .

Use image:

```bash
docker run --rm -it --gpus all darts-pytorch  # just remove  `--gpus all` when using CPU
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# should print (0.3.1, True)

./darts_ref/run_darts_inference.sh
```

Mount current host directory as `host_files` in container:

```bash
docker run --rm -it --gpus all -v $(pwd):/workdir/host_files darts-pytorch
```
