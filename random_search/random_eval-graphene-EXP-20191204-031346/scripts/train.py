import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR, NetworkGalaxyZoo
from model_search import Network # for random search
from datasets import load_dataset, DSET_NAME_TBL
from genotypes import GENOTYPE_TBL
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser("darts")

# NON-PERFORMANCE RELATED
parser.add_argument('--dataset', type=str, default='cifar', help='name of the dataset to use (e.g. cifar, mnist, graphene)')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--folder_name', type=str, default=None, help='Name of top-level folder containing the data, e.g. galaxy_zoo')
parser.add_argument('--use_xarray', action='store_true', default=True, help='use xarray package for data loading')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--random', action="store_true", default=False, help='train a random cell')

# PERFORMANCE RELATED

## HIGHLY IMPORTANT
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability') # this is dropout
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--arch', type=str, default='DATASET',
                    help='which architecture to use; default is lookup by dataset name')

## MEDIUM IMPORTANT
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer; one of SGD or Adam')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--fc1_size', type=int, default=1024, help='number of units in fully connected layer 1')
parser.add_argument('--fc2_size', type=int, default=1024, help='number of units in fully connected layer 2')
parser.add_argument('--gz_dtree', action='store_true', default=False,
                    help='run GalaxyZoo with decision tree structure (default False: use simple regression)')
parser.add_argument('--primitives', type=str, default='Default',
                    help='set of primitive operations for arch search; defined in genotypes.py')

## LESS IMPORTANT
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of validation data')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

args = parser.parse_args()

args.save = 'eval-{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
if args.random:
  args.save = 'random_' + args.save
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Get normalized dataset name
dataset = DSET_NAME_TBL[args.dataset.lower().strip()]

# If the default set of primitives is requested, use the normalized name of the dataset
primitives_name = dataset if args.primitives == 'Default' else args.primitives

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True

  if not args.random:
    # We would always get the same random architecture if we set the random
    # seed here. We'll set it after finding a random genotype.
    torch.manual_seed(args.seed)

  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  train_data, OUTPUT_DIM, IN_CHANNELS, is_regression = load_dataset(args, train=True)

  criterion = nn.CrossEntropyLoss() if not is_regression else nn.MSELoss()

  if args.random:
    model_tmp = Network(C=args.init_channels, num_classes=OUTPUT_DIM, layers=args.layers, 
                        primitives_name=primitives_name, criterion=criterion, num_channels=IN_CHANNELS)
    genotype = model_tmp.genotype()  # Random

    # We can now set the random seed.
    torch.manual_seed(args.seed)
  # If the architecture was the default DATASET, look up the architecture corresponding to this dataset
  elif args.arch == 'DATASET':
    genotype = GENOTYPE_TBL[dataset]
    print(f'using genotype for {dataset}')
  else:
    try:
      genotype = eval("genotypes.%s" % args.arch)
    except (AttributeError, SyntaxError):
      genotype = genotypes.load_genotype_from_file(args.arch)

  genotypes.save_genotype_to_file(genotype, os.path.join(args.save, "genotype.arch"))
  # Set the inference network; default is NetworkCifar10; supported alternatives NetworkGalaxyZoo
  if dataset == 'GalaxyZoo' and args.gz_dtree:
    model = NetworkGalaxyZoo(C=args.init_channels, num_classes=OUTPUT_DIM, layers=args.layers, genotype=genotype,
                             fc1_size=args.fc1_size, fc2_size=args.fc2_size, num_channels=IN_CHANNELS)
  else:
    model = NetworkCIFAR(args.init_channels, OUTPUT_DIM, args.layers, args.auxiliary, genotype, num_channels=IN_CHANNELS)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = criterion.cuda()

  # build optimizer based on optimizer input; one of SGD or Adam
  if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  elif args.optimizer == 'Adam':
    optimizer= torch.optim.Adam(
    params=model.parameters(),
    lr=args.learning_rate,
    betas=(0.90, 0.999),
    weight_decay=args.weight_decay)
  else:
    raise ValueError(f"Bad optimizer; got {args.optimizer}, must be one of 'SGD' or 'Adam'.")

  # Split training data into training and validation queues
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  # history of training and validation loss; 2 columns for loss and accuracy / R2
  hist_trn = np.zeros((args.epochs, 2))
  hist_val = np.zeros((args.epochs, 2))
  metric_name = 'accuracy' if not is_regression else 'R2'

  for epoch in range(args.epochs):
    # scheduler.step()
    # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    logging.info('epoch %d lr %e', epoch, args.learning_rate)
    # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    model.drop_path_prob = args.drop_path_prob

    # training results
    train_acc, train_obj = train(train_queue, model, criterion, optimizer, is_regression=is_regression)
    logging.info(f'training loss; {metric_name}: {train_obj:e} {train_acc:f}')
    # save history to numpy arrays
    hist_trn[epoch] = [train_acc, train_obj]
    np.save(os.path.join(args.save, 'hist_trn'), hist_trn)

    # validation results
    valid_acc, valid_obj = infer(valid_queue, model, criterion, is_regression=is_regression)
    logging.info(f'validation loss; {metric_name}: {valid_obj:e} {valid_acc:f}')
    # save history to numpy arrays
    hist_val[epoch] = [valid_acc, valid_obj]
    np.save(os.path.join(args.save, 'hist_val'), hist_val)

    # save current model weights
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer, is_regression=False):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    n = input.size(0)

    if not is_regression:
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    else:
        r2 = r2_score(target.data.cpu().numpy(), logits.data.cpu().numpy())
        objs.update(loss.data[0], n)
        top1.update(r2, n) # "top1" for regression is the R^2

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, is_regression=False):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    n = input.size(0)

    if not is_regression:
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    else:
        r2 = r2_score(target.data.cpu().numpy(), logits.data.cpu().numpy())
        objs.update(loss.data[0], n)
        top1.update(r2, n) # "top1" for regression is the R^2

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
