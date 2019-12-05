## Datasets Module

import os
import utils
import torch
import torchvision.datasets as dset
from torch.utils.data import TensorDataset, Dataset
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
import boto3

from galaxy_zoo import DatasetGalaxyZoo

VALID_DSET_NAMES = {
    'CIFAR': ['cifar', 'cifar10', 'cifar-10'],
    'MNIST': ['mnist'],
    'FashionMNIST': ['fashionmnist', 'fashion-mnist', 'mnistfashion'],
    'GrapheneKirigami': ['graphene', 'graphenekirigami', 'graphene-kirigami', 'kirigami'],
    'GalaxyZoo': ['galaxy-zoo', 'galaxyzoo'],
    'ChestXRay': ['chest-xray', 'chest-x-ray']
}

# Table to normalize data set name; key = aliased name, value = canonical dataset name
DSET_NAME_TBL = dict()
for dset_name_norm, dset_alias_list in VALID_DSET_NAMES.items():
    for dset_alias_name in dset_alias_list:
        DSET_NAME_TBL[dset_alias_name] = dset_name_norm

# AWS storage bucket
BUCKET_NAME = 'capstone2019-google'

# *****************************************************************************
def load_dataset(args, train=True):
    """ function to load datasets (e.g. CIFAR10, MNIST, FashionMNIST, Graphene)

    input:
        args - result of ArgumentParser.parse() containing a `dataset` property

    output:
        data - a torch Dataset
        output_dim - an integer representing necessary output dimension for a model
        # is_regression - boolean indicator for regression problems
        inference_type - one of the following three strings:
            'classification': model predicts one of N classes, encoded with one-hot classification; e.g.CIFAR-10
            'regression': model predicts N real valued outputs; loss is mean squared error, e.g. galaxy-zoo
            'multi_binary': model predicts N independent binary class labels (0 or 1)
    """
    dset_name = args.dataset.lower().strip()

    if dset_name in VALID_DSET_NAMES['CIFAR']:
        # from the original DARTS code
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        tr = train_transform if train else valid_transform
        data = dset.CIFAR10(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 3
        # is_regression = False
        inference_type = 'classification'

    elif dset_name in VALID_DSET_NAMES['MNIST']:
        train_transform, valid_transform = utils._data_transforms_mnist(args)
        tr = train_transform if train else valid_transform
        data = dset.MNIST(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 1
        # is_regression = False
        inference_type = 'classification'

    elif dset_name in VALID_DSET_NAMES['FashionMNIST']:
        train_transform, valid_transform = utils._data_transforms_mnist(args)
        tr = train_transform if train else valid_transform
        data = dset.FashionMNIST(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 1
        # is_regression = False
        inference_type = 'classification'

    elif dset_name in VALID_DSET_NAMES['GrapheneKirigami']:
        # load xarray dataset
        data_path = os.path.join(args.data, 'graphene_processed.nc')
        ds = xr.open_dataset(data_path)

        # X = ds['coarse_image'].values  # coarse 3x5 image (not using it)
        X = ds['fine_image'].values  # the same model works worse on higher resolution image
        y = ds['strain'].values
        X = X[..., np.newaxis]  # add channel dimension
        y = y[:, np.newaxis]  # pytorch wants ending 1 dimension

        # pytorch conv2d wants channel-first, unlike Keras
        X = X.transpose([0, 3, 1, 2])  # (sample, x, y, channel) -> (sample, channel, x, y)

        # it appears we need each dimension to be twice divisible by 2
        # reshape from 30x80 -> 32x80 by zero-padding
        # see here for details https://stackoverflow.com/a/46115998
        X = np.pad(X, [(0, 0), (0, 0), (1, 1), (0, 0)], mode='constant', constant_values=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if train:
            data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        else:
            data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        output_dim = 1
        in_channels = 1
        # is_regression = True
        inference_type = 'regression'

    elif dset_name in VALID_DSET_NAMES['GalaxyZoo']:
        if not args.use_xarray:
            # parent path for the galaxy zoo dataset
            if args.folder_name is not None:
                data_path = os.path.join(args.data, args.folder_name)
            else:
                data_path = os.path.join(args.data, 'galaxy_zoo')
            # train and test transforms for this data set; choose applicable transform
            train_transform, valid_transform = utils._data_transforms_galaxy_zoo(args)
            transform = train_transform if train else valid_transform
            # the locations of the images and CSV label files for train and test
            train_img_dir = os.path.join(data_path, 'images_train')
            train_csv_file = os.path.join(data_path, 'labels_train/labels_train.csv')
            # for test data, only images available; the labels are NOT ground truth, just a placeholder!
            test_img_dir = os.path.join(data_path, 'images_test')
            test_csv_file = os.path.join(data_path, 'benchmark_solutions/central_pixel_benchmark.csv')
            # choose appropriate image directory and CSV file
            csv_file = train_csv_file if train else val_csv_file
            # instantiate the Dataset
            if train:
                data = DatasetGalaxyZoo(train_img_dir, train_csv_file, transform=transform)
            else:
                data = DatasetGalaxyZoo(test_img_dir, test_csv_file, transform=transform)
        else:
            if train:
                if args.folder_name is not None:
                    ds = xr.open_dataset(os.path.join(args.data, args.folder_name, 'galaxy_train.nc'))
                else:
                    ds = xr.open_dataset(os.path.join(args.data, 'galaxy_train.nc'))
                X = ds['image_train'].transpose('sample', 'channel', 'x' ,'y').data  # pytorch use channel-first, unlike Keras
                y = ds['label_train'].data
            else:
                # ds = xr.open_dataset(os.path.join(args.data, args.folder_name, 'galaxy_test.nc'))
                raise NotImplementedError('Test loading with xarray not implemented')

            try:
                import torchsample
            except:
                raise RuntimeError('Install torchsample: pip install git+https://github.com/ncullen93/torchsample')
                
            # tips from http://benanne.github.io/2014/04/05/galaxy-zoo.html
            transform = torchsample.transforms.Compose([
                torchsample.transforms.Rotate(90),
                torchsample.transforms.RandomFlip()
            ])

            data = torchsample.TensorDataset(
                torch.from_numpy(X), torch.from_numpy(y),
                input_transform = transform
            )

        # these parameters don't depend on train vs. validation
        output_dim = 37
        in_channels = 3
        # is_regression = True
        inference_type = 'regression'

    elif dset_name in VALID_DSET_NAMES['ChestXRay']:
        if train:
            if args.folder_name is not None:
                ds = xr.open_dataset(os.path.join(args.data, args.folder_name, 'chest_xray.nc'))
            else:
                ds = xr.open_dataset(os.path.join(args.data, 'chest_xray.nc'))
                
            # pytorch use channel-first, unlike Keras; order is (sample, channel, x, y)
            X = ds['image'].transpose('sample', 'x' ,'y').data
            # add channel dimension 
            X = np.expand_dims(X, axis=1)
            y = ds['label'].transpose('sample', 'feature').data
        else:
            # ds = xr.open_dataset(os.path.join(args.data, args.folder_name, 'galaxy_test.nc'))
            raise NotImplementedError('Test loading with xarray not implemented')

        # Convert numpy arrays to torch tensors
        X_torch = torch.from_numpy(X)
        # labels have data type int8 in Xarray / Numpy; need to convert to uint8 for pytorch
        y_torch = torch.from_numpy(y.astype(np.float32))
        data = TensorDataset(
            X_torch, y_torch
        )

        # these parameters don't depend on train vs. validation
        # 14 different diseases; each can be labeled 0 or 1 independently (not n-fold classification!)
        output_dim = 14
        in_channels = 1
        # TODO: use linear regression as placeholder; need to switch it to logistic regression semantics
        # is_regression = True
        inference_type = 'multi_binary'

    else:
        exc_str = 'Unable to match provided dataset name: {}'.format(dset_name)
        exc_str += '\nValid names are case-insensitive elements of: {}'.format(VALID_DSET_NAMES)
        raise RuntimeError(exc_str)

    return data, output_dim, in_channels, inference_type
