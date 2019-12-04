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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network, NetworkGalaxyZoo
from architect import Architect
from datasets import load_dataset, DSET_NAME_TBL
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
parser.add_argument('--seed', type=int, default=2, help='random seed')

# PERFORMANCE-RELATED
### Highly important
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-4, help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--L1_lambda', type=float, default=None, help='multiplier on the architecture weight L1 penalty')
parser.add_argument('--arch_learning_rate', type=float, default=1e-2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-6, help='weight decay for arch encoding')
parser.add_argument('--cell_steps', type=int, default=4, help='number of steps in a cell (see model_search.Network)')
parser.add_argument('--cell_multiplier', type=int, default=4, help='multiplier for a cell (see model_search.Network)')
### Somewhat important
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer; one of SGD or Adam')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gz_dtree', action='store_true', default=False,
                    help='run GalaxyZoo with decision tree structure (default False: use simple regression)')
parser.add_argument('--fc1_size', type=int, default=1024, help='for gz_dtree: number of units in fully connected layer 1')
parser.add_argument('--fc2_size', type=int, default=1024, help='for gz_dtree: number of units in fully connected layer 2')
parser.add_argument('--primitives', type=str, default='Default',
                    help='set of primitive operations for arch search; defined in genotypes.py')
### Not very important
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# the below arg is not used anywhere in this script, nor is it used in any of the loaded modules
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  train_data, OUTPUT_DIM, IN_CHANNELS, inference_type = load_dataset(args, train=True)

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

  if inference_type == 'classification':
    criterion = nn.CrossEntropyLoss()
  elif inference_type == 'regression':
    criterion = nn.MSELoss()
  elif inference_type == 'multi_binary':
    criterion = nn.BCEWithLogitsLoss()
  else:
    raise ValueError("Bad inference_type; must be one of classification, regression, or multi_binary")

  criterion = criterion.cuda()

  # Special network for Galaxy Zoo regression
  if dataset == 'GalaxyZoo' and args.gz_dtree:
      model = NetworkGalaxyZoo(C=args.init_channels, num_classes=OUTPUT_DIM, primitives_name=primitives_name,
                             layers=args.layers, criterion=criterion,
                             fc1_size=args.fc1_size, fc2_size=args.fc2_size,
                             num_channels=IN_CHANNELS, steps=args.cell_steps,
                             multiplier=args.cell_multiplier)
  else:
      model = Network(C=args.init_channels, num_classes=OUTPUT_DIM, primitives_name=primitives_name,
                      layers=args.layers, criterion=criterion, num_channels=IN_CHANNELS,
                      steps=args.cell_steps, multiplier=args.cell_multiplier)

  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args, L1_lambda=args.L1_lambda)

  # history of training and validation loss; 2 columns for loss and accuracy / R2
  hist_trn = np.zeros((args.epochs, 2))
  hist_val = np.zeros((args.epochs, 2))
  metric_name = 'accuracy' if inference_type == 'classification' else 'R2'

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # saving & logging weights for architecture
    normal_weights = F.softmax(model.alphas_normal, dim=-1)
    reduce_weights = F.softmax(model.alphas_reduce, dim=-1)
    logging.info('\nalphas_normal = {}'.format(normal_weights))
    logging.info('\nalphas_reduce = {}'.format(reduce_weights))
    np.save(os.path.join(args.save, 'normal_%03d' % epoch), normal_weights.data.cpu().numpy())
    np.save(os.path.join(args.save, 'reduce_%03d' % epoch), reduce_weights.data.cpu().numpy())

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, inference_type=inference_type)
    logging.info(f'training loss; {metric_name}: {train_obj:e} {train_acc:f}')
    # save history to numpy arrays
    hist_trn[epoch] = [train_acc, train_obj]
    np.save(os.path.join(args.save, 'hist_trn'), hist_trn)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, inference_type=inference_type)
    logging.info(f'validation loss; {metric_name}: {valid_obj:e} {valid_acc:f}')
    # save history to numpy arrays
    hist_val[epoch] = [valid_acc, valid_obj]
    np.save(os.path.join(args.save, 'hist_val'), hist_val)

    # save weights
    utils.save(model, os.path.join(args.save, 'weights.pt'))

    # save loss history


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, inference_type='classification'):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if inference_type == 'classification':
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    elif inference_type in ['regression', 'multi_binary']:
        r2 = r2_score(target.data.cpu().numpy(), logits.data.cpu().numpy())
        objs.update(loss.data[0], n)
        top1.update(r2, n) # "top1" for regression is the R^2

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f', step, objs.avg, top1.avg)
<<<<<<< Updated upstream
=======
    elif inference_type == 'multi_binary':
        objs.update(loss.data[0], n)
        top1.update(auc, n) # "top1" for binary classification is the AUC

        if step % args.report_freq == 0:
          logging.info('train %03d %e %f', step, objs.avg, top1.avg)
>>>>>>> Stashed changes
    else:
        ValueError

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, inference_type='classification'):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    n = input.size(0)

    logits = model(input)
    loss = criterion(logits, target)

    if inference_type == 'classification':
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    elif inference_type in ['regression', 'multi_binary']:
        # use the same metric for both cases for now
        r2 = r2_score(target.data.cpu().numpy(), logits.data.cpu().numpy())
        objs.update(loss.data[0], n)
        top1.update(r2, n) # "top1" for regression is the R^2

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
<<<<<<< Updated upstream
=======
    elif inference_type == 'multi_binary':
        print(target.data.cpu().numpy().shape)
        print(logits.data.cpu().numpy().shape)
        print(target.data.cpu().numpy()[:5, :5])
        print(logits.data.cpu().numpy()[:5, :5])
        auc = roc_auc_score(target.data.cpu().numpy(),
                            sigmoid(logits.data.cpu().numpy())
                            )
        objs.update(loss.data[0], n)
        top1.update(auc, n) # "top1" for binary classification is the AUC

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
>>>>>>> Stashed changes
    else:
        ValueError

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
