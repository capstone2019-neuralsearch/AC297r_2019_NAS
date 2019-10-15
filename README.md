# AC297r_2019_NAS
Harvard IACS Data Science Capstone: Neural Architecture Search (NAS) with Google

## DARTS Code

The [DARTS code](https://github.com/capstone2019-neuralsearch/darts) we are using is a fork of the original code by Hanxiao Liu (quark0), which we have extended to fit our needs.

## Python Environment

### Use Nvidia-Docker

The DARTS code requires a specific version of PyTorch and CUDA, which could be difficult to install on some machines. Thus we recommend using [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) to easily replicate the same environment on different platforms (multiple cloud vendors and local servers). Please follow [docker/install-nvidia-docker](./docker/install-nvidia-docker) to install Nvidia-Docker on a cloud GPU instance, and follow [docker/darts-pytorch-image](./docker/darts-pytorch-image) to use the our `darts-pytorch` image.

Alternatively, you can install the libraries without Docker, as shown below.

### Install on native environment

We start with the original DARTS code available here: https://github.com/quark0/darts

This code has been verified on the following Python environment:

* Python = 3.6 (version 3.7 causes syntax error due to ``async`` keyword)
* torch = 0.3.1
* torchvision = 0.2.0
* CUDA = 10.0
* operating system: Ubuntu Linux 18.04 LTS

To create an Anaconda environment, run the following commands in Ubuntu 18.04:
```
$ conda create -n darts python=3.6
$ conda activate darts
(darts) $ conda install cudatoolkit=10.0
(darts) $ pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
(darts) $ pip install torchvision==0.2.0
```
This looks a bit odd.  We are mixing Anaconda and pip, which is not usually recommended.  Here we need a slightly unusual configuration.  The code is running on a GPU server with newer GPU cards that seem to require CUDA version 10.0 to run.  But the old version of Pytorch has a requirement that CUDA = 9.0.  We work around this by first installing CUDA 10.0 with Anaconda, then using a lower level pip installation of torch and torchvision.  It turns out that CUDA 10.0 is backward compatible enough with CUDA 9.0 that the wheel for torch still works.

A few other Anaconda packages must also be installed.  These are all routine.

```
(darts) $ conda install scipy scikit-learn graphviz xarray boto3 python-graphviz pydot
```

## Running Architecture Search

The first step in building a new model is to run the architecture search with DARTS.

```
$ cd darts/cnn
$ conda activate darts
(darts) $ python train_search.py --dataset cifar-10 --save CIFAR-10 --gpu 1
(darts) $ python train_search.py --dataset mnist --save MNIST --gpu 2
(darts) $ python train_search.py --dataset fashion-mnist --save FASHION_MNIST --gpu 3
(darts) $ python train_search.py --dataset graphene --save GRAPHENE --batch_size 24 --gpu 1
```
The first command runs an architecture search on the CIFAR-10 dataset.  It saves the results into a directory named e.g. 'search-CIFAR-10-2019104-141028'.  The CIFAR-10 is the --save argument; the rest is YYYYMMDD-HHMMSS.  The search uses GPU device 1.
The second command runs an architecture search on the MNIST digits dataset on GPU device  2.
The third command runs an architecture search on the FASHION-MNIST dataset on GPU device 3.
The fourth command runs an architecture search on the graphene kirigami dataset on GPU device 1.  The batch size must be manually reduced from the default of 64 because the input image size of 30x80 is larger than on the first 3 data sets and otherwise the GPU runs out of memory.

These searches took approximately 17 hours on a high end NVIDIA RTX 2080 Ti GPU.  The authors reported architecture searches took slightly longer (about one day) on NVIDIA GTX 1080 Ti GPUs, which were high end GPUs a few years ago.

There are two outputs with the results of the search in each directory.  The first is PyTorch weights file names weights.pt.  The second is a log called log.txt.  The log includes a history of the training and validation loss on the best cell.  It also periodically prints out a grid with the current "attention" weights on the different possible cells.  Most importantly, it periodically logs a line of Python that can be used to define a genotype in the file genotypes.py.  We will need this in the next step!

## Training a Network with the Discovered Architecture

Once the best architecture is discovered, it needs to be made available for training a full network.  Currently the process for doing this is a bit manual.  Take the MNIST training as an example.  Go into the directory e.g. darts/cnn/search-CIFAR-10-20191004-160833.  In this directory, open up log.txt.  At the very bottom of this file (line 1203) you will see that the final validation accuracy is 99.07%, which is promising.  A few lines up, on line 1181, is the final extract with the genotype.  After the timestamp, this line reads

`genotype = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))`

Copy the fragment above starting with ``genotype =  ``.  Now open the file `genotypes.py`.  At the bottom of this file, we will define a new variable.  We follow the naming convention and use all caps.  I chose the name MNIST for this variable, because we're just running one architecture search on each data set for now.  I add a line `MNIST = ...` where `MNIST` replaces `genotype` in the fragment pasted from the other file.   In the next part, we are going to train a new network from scratch using this architecture, but with more cells.  This is done with the program `train.py`  The argument ``--arch MNIST`` tells the program to get the genotype instance named MNIST in the file genotypes.py.  Here are the commands to run the training on CIFAR-10, MNIST and FASHION-MNIST:

```
$ cd darts/cnn
$ conda activate darts
(darts) $ python train.py --dataset cifar-10 --arch CIFAR_10 --save CIFAR-10 --gpu 1
(darts) $ python train.py --dataset mnist --arch MNIST --save MNIST --gpu 2
(darts) $ python train.py --dataset fashion-mnist --arch FASHION_MNIST --save FASHION-MNIST --gpu 3
(darts) $ python train.py --dataset graphene --arch GRAPHENE --save GRAPHENE --batch_size 32 --gpu 1
(darts) $ python train.py --dataset graphene --arch GRAPHENE --save GRAPHENE --layers 8 --gpu 1
```

The second command trains a full network on the MNIST dataset.  It uses the architecture (Genotype instance) named MNIST in the file genotypes.py.  This is the variable we copy / pasted from the log file in the architecture search to the bottom of genotypes.py.  The `--save` argument puts the output into a directory named e.g. `eval-MNIST-20191005-150239`.  The `--gpu 2` argument runs the job on GPU 2.  (You can run as many parallel jobs as you have GPUs on your machine; the training in GPU bound.)

Training the full network for CIFAR-10, MNIST, and FASHION-MNIST each took approximately 26 hours on an NVIDIA RTX 2080 Ti GPU.

The first attempt to train the graphene model uses the default number of layers, 20.  This led to an overfit model with negative R2 on both train and validation data.  The second call uses 8 layers, the same as was used in the architecture search.

The outputs of this training process are the same as for architecture search.  In the eval directory there will be two output files: a log file called `log.txt` and a PyTorch model weights file called `weights.pt`.  For our project, we created a directory called `models` under the top level directory `darts` (`darts/models`) where we could save all of our trained models.  Copy / paste `models.pt` from the MNIST training to this directory with the file name `mnist_model.pt`.  We download the original model the authors trained on CIFAR-10, and saved it here as `cifar10_model_original.pt`.  We compare it to the new version we trained from scratch called `cifar10_model.pt`.

## Testing Model Performance

After the trained models are copied into the `models` directory, we can use `test.py` to test their performance on held out test data.  Usually we have a pretty good idea of what the performance will be by looking at the validation performance on the last epoch of training, but of course there is no substitute for a held out test set.  Here is the command to test the original model the authors built for CIFAR-10:

``python test.py --auxiliary --dataset cifar-10 --arch CIFAR_10 --model_path ../models/cifar10_model_original.pt``

This runs very quickly and shows a test accuracy of 97.37%.

Here is the evaluation for our version of the CIFAR-10 model, trained from scratch with our discovered architecture:

``python test.py --auxiliary --dataset cifar-10 --arch CIFAR_10 --model_path ../models/cifar10_model.pt``

This has a test accuracy of 97.10%.  It's slightly worse than the authors, but they did multiple training runs with their discovered cell and reported the best one.  We trained just one.

Here are the commands to test our model on MNIST and FASHION-MNIST.

```
$ cd darts/cnn
$ conda activate darts
(darts) $ python test.py --dataset mnist --arch MNIST --model_path ../models/mnist_model.pt
(darts) $ python test.py --dataset fashion-mnist --arch FASHION_MNIST --model_path ../models/fashion_mnist_model.pt
(darts) $ python test.py --dataset graphene --arch GRAPHENE --layers 8 --model_path ../models/graphene_model.pt 
```

Our MNIST model has a classification accuracy of 99.28% on test data.

Our fashion-mnist model has a classification accuracy of 99.29% on test data.

Our graphene model has a regression R<sup>2</sup> of 0.9085.

## Graphene Kirigami Dataset

To download the processed dataset our code loads for the Graphene Kirigami problem, run:

```bash
wget https://capstone2019-google.s3.amazonaws.com/graphene_processed.nc -P /path/to/save
python train_search.py --data /path/to/save --dataset graphene
```

## Presentations

Lightning Talk 1: [link](https://docs.google.com/presentation/d/e/2PACX-1vQ2CSXrC6_XlA7eqp5wvdu1_ysZBthoF0uy5pUgR56WguDSWM_7ye34qAEf71YOFTLxAHyNtRl4nt9P/pub?start=false&loop=false&delayms=30000)

Milestone 1: https://docs.google.com/presentation/d/101dRBnm_5-AD_yxL1krIotfki2spr-QjvJtJrKkLpgY/edit#slide=id.g61d1bae56f_0_547

Milestone 2:  https://docs.google.com/presentation/d/1ETMiDSIVxW6pK-f69YLRunq9HoD08kR7TJadng80n2w/edit?ts=5da3d4db#slide=id.g6230f468c2_0_13 

