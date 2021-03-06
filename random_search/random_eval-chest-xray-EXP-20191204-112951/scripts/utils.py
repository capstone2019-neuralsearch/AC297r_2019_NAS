import os
import numpy as np
import torch
import torch.nn as nn
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

def loss_criterion(inference_type: str):
  """
  Return the loss function (criterion) suitable to this inference type.
  INPUTS:
    inference_type: one of 'classification', 'regression', or 'multi_binary'
  OUTPUTS:
    criterion: a torch function to compute the loss
  """
  if inference_type == 'classification':
    criterion = nn.CrossEntropyLoss()
  elif inference_type == 'regression':
    criterion = nn.MSELoss()
  elif inference_type == 'multi_binary':
    criterion = nn.BCEWithLogitsLoss()
  else:
    raise ValueError("Bad inference_type; must be one of classification, regression, or multi_binary")
  # Return the cuda version of this function
  return criterion.cuda()

class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_mnist(args):
  MNIST_MEAN = [33.318421449829934]
  MNIST_STD = [78.56748998339798]

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_galaxy_zoo(args):
    """Data transformations for the Galaxy Zoo data set"""
    # Parameters for this data set; mean, std dev, original image size, and new image size
    GZ_MEAN = [0.485, 0.456, 0.406]
    GZ_STD = [0.229, 0.224, 0.225]
    CENTER_SIZE = 224
    SHRUNKEN_SIZE = 56
    # training transform includes resize; random flips and rotations; and normalization
    train_transform = transforms.Compose([
        transforms.CenterCrop((CENTER_SIZE, CENTER_SIZE)),
        transforms.Resize((SHRUNKEN_SIZE, SHRUNKEN_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0,360)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=GZ_MEAN, std=GZ_STD)  
    ])
    # validation transform includes only resize and normalization
    valid_transform = transforms.Compose([
        transforms.CenterCrop((CENTER_SIZE, CENTER_SIZE)),
        transforms.Resize((SHRUNKEN_SIZE, SHRUNKEN_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=GZ_MEAN, std=GZ_STD)  
    ])
    return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
