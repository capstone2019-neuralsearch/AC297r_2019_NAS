# AC297r_2019_NAS
Harvard IACS Data Science Capstone: Neural Architecture Search (NAS) with Google

## Python Environment

We start with the original DARTS code available here: https://github.com/quark0/darts

This code has been verified on the following Python environment:

* Python = 3.6 (version 3.7 causes syntax error due to ``async`` keyword)
* torch = 0.3.1
* torchvision = 0.2.0
* CUDA = 10.0
* operating system: Ubuntu Linux (18.04 LTS recommended)

To create an Anaconda environment, run the following commands in Ubuntu 18.04:
```
$ conda create -n darts python=3.6
$ conda activate darts
(darts) $ conda install cudatoolkit=10.0
(darts) $ pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
(darts) $ pip install torchvision==0.2.0
```
This looks a bit odd.  We are mixing Anaconda and pip, which is not usually recommended.  Here we need a slightly unusual configuration.  The code is running on a GPU server with newer GPU cards that seem to require CUDA version 10.0 to run.  But the old version of Pytorch has a requirement that CUDA = 9.0.  We work around this by first installing CUDA 10.0 with Anaconda, then using a lower level pip installation of torch and torchvision.  It turns out that CUDA 10.0 is backward compatible enough with CUDA 9.0 that the wheel for torch still works.

## Presentations

Lightning Talk 1: [link](https://docs.google.com/presentation/d/e/2PACX-1vQ2CSXrC6_XlA7eqp5wvdu1_ysZBthoF0uy5pUgR56WguDSWM_7ye34qAEf71YOFTLxAHyNtRl4nt9P/pub?start=false&loop=false&delayms=30000)

Lightning Talk 2: https://docs.google.com/presentation/d/101dRBnm_5-AD_yxL1krIotfki2spr-QjvJtJrKkLpgY/edit#slide=id.g61d1bae56f_0_547

