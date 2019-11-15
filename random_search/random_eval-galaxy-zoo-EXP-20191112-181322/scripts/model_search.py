import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
# from genotypes import PRIMITIVES
from genotypes import PRIMITIVES_TBL
from genotypes import Genotype

class MixedOp(nn.Module):

  def __init__(self, C, stride, primitives_name: str):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    # Look up collection of primitives
    self.PRIMITIVES = PRIMITIVES_TBL[primitives_name]
    for primitive in self.PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, primitives_name: str, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.PRIMITIVES = PRIMITIVES_TBL[primitives_name]
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C=C, stride=stride, primitives_name=primitives_name)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, primitives_name: str, criterion, num_channels=3, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._primitives_name = primitives_name
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._num_channels = num_channels
    self.PRIMITIVES = PRIMITIVES_TBL[primitives_name]

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      # in_channels, out_channels, kernel_size
      nn.Conv2d(self._num_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(primitives_name=primitives_name, steps=steps, multiplier=multiplier,
                  C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction=reduction, reduction_prev=reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(C=self._C, num_classes=self._num_classes, layers=self._layers, primitives_name=self._primitives_name,
                        criterion=self._criterion, num_channels=self._num_channels, steps=self._steps,
                        multiplier=self._multiplier).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES_TBL[self._primitives_name])

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != self.PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((self.PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


class NetworkGalaxyZoo(Network):
  """Subclass of Network specialized for the GalaxyZoo problem"""

  def __init__(self, primitives_name: str, C, num_classes, layers, criterion,
               fc1_size: int, fc2_size: int, num_channels=3, steps=4, multiplier=4, stem_multiplier=3):
    # super(Network, self).__init__()
    # Network.__init__(self, C=C, num_classes=num_classes, layers=layers, primitives_name=primitives_name,
    #                 criterion=criterion, num_channels=num_channels)
    nn.Module.__init__(self)
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._primitives_name = primitives_name
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._num_channels = num_channels
    self.PRIMITIVES = PRIMITIVES_TBL[primitives_name]

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      # in_channels, out_channels, kernel_size
      nn.Conv2d(self._num_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(primitives_name=primitives_name, steps=steps, multiplier=multiplier,
                  C_prev_prev=C_prev_prev, C_prev=C_prev, C=C_curr, reduction=reduction, reduction_prev=reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)

    # Fully connected layers
    self.fc1 = nn.Linear(C_prev, fc1_size)
    self.fc2 = nn.Linear(fc1_size, fc2_size)
    fc_last_size = fc1_size

    # Galaxy Zoo question 1: smooth galaxy; galaxy with features or disk; elliptic
    self.classifier_q1 = nn.Linear(fc_last_size, 3)

    # Galaxy Zoo question 2: Is it edge on?
    self.classifier_q2 = nn.Linear(fc_last_size, 2)

    # Galaxy Zoo question 3: Is there a bar?
    self.classifier_q3 = nn.Linear(fc_last_size, 2)

    # Galaxy Zoo question 4: Is there a spiral pattern?
    self.classifier_q4 = nn.Linear(fc_last_size, 2)

    # Galaxy Zoo question 5: How prominent is the central bulge?
    self.classifier_q5 = nn.Linear(fc_last_size, 4)

    # Galaxy Zoo question 6: Is there anything odd?
    self.classifier_q6 = nn.Linear(fc_last_size, 2)

    # Galaxy Zoo question 7: How rounded is it?
    self.classifier_q7 = nn.Linear(fc_last_size, 3)

    # Galaxy Zoo question 8: What is the odd feature?
    self.classifier_q8 = nn.Linear(fc_last_size, 7)

    # Galaxy Zoo question 9: Is Does the galaxy have a bulge?
    self.classifier_q9 = nn.Linear(fc_last_size, 3)

    # Galaxy Zoo question 10: How tightly wound is it?
    self.classifier_q10 = nn.Linear(fc_last_size, 3)

    # Galaxy Zoo question 11: How many spiral arms?
    self.classifier_q11 = nn.Linear(fc_last_size, 6)

    # initialize edge weights
    self._initialize_alphas()

  def new(self):
    model_new = NetworkGalaxyZoo(
        C=self._C, num_classes=self._num_classes, layers=self._layers, primitives_name=self._primitives_name,
        criterion=self._criterion, num_channels=self._num_channels).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    """
    Specialized forward pass for GalaxyZoo; hybrid of classification and regression.
    The first part of the forward pass is the same as
    """
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)

    # Fully connected layers
    conv_out = out.view(out.size(0),-1)
    fc1_out = self.fc1(conv_out)
    fc2_out = self.fc2(fc1_out)
    fc_out = fc2_out

    # Logits for classifiers on GalaxyZoo questions
    logits_q1 = self.classifier_q1(fc_out)
    logits_q2 = self.classifier_q2(fc_out)
    logits_q3 = self.classifier_q3(fc_out)
    logits_q4 = self.classifier_q4(fc_out)
    logits_q5 = self.classifier_q5(fc_out)
    logits_q6 = self.classifier_q6(fc_out)
    logits_q7 = self.classifier_q7(fc_out)
    logits_q8 = self.classifier_q8(fc_out)
    logits_q9 = self.classifier_q9(fc_out)
    logits_q10 = self.classifier_q10(fc_out)
    logits_q11 = self.classifier_q11(fc_out)

    # Classification probabilities for GalaxyZoo questions
    # Each output is a product of (probability classification is relevant) x (conditional probabilities)
    # A1 = C1
    probs_q1 = F.softmax(logits_q1, dim=-1)
    C1_1 = probs_q1[:,0:1]
    C1_2 = probs_q1[:,1:2]

    # alternative method: follow Dieleman and use relu followed by "divisive normalization"
    # this makes it easier for model to predict hard zeros
    # normalizer = 1.0E-12
    # logits_q1 = F.relu(logits_q1) + normalizer
    # probs_q1 = logits_q1 / (torch.sum(logits_q1, dim=-1).view(-1,1))

    # A2 = C1.2 * C2
    probs_q2 = C1_2 * F.softmax(logits_q2, dim=-1)
    C2_1 = probs_q2[:,0:1]
    C2_2 = probs_q2[:,1:2]

    # A3 = C2.2 * C3
    probs_q3 = C2_2 * F.softmax(logits_q3, dim=-1)

    # A4 = C2.2 * C4
    probs_q4 = C2_2 * F.softmax(logits_q4, dim=-1)
    C4_1 = probs_q4[:,0:1]

    # A5 = C2.2 * C5
    probs_q5 = C2_2 * F.softmax(logits_q5, dim=-1)

    # A6 = C6
    probs_q6 = F.softmax(logits_q6, dim=-1)
    C6_1 = probs_q1[:,0:1]

    # A7 = C1.1 * C7
    probs_q7 = C1_1 * F.softmax(logits_q7, dim=-1)

    # A8 = C6.1 * C8
    probs_q8 = C6_1 * F.softmax(logits_q8, dim=-1)

    # A9 = C2.1 * C9
    probs_q9 = C2_1 * F.softmax(logits_q9, dim=-1)

    # A10 = C4.1 * C10
    probs_q10 = C4_1 * F.softmax(logits_q10, dim=-1)

    # A11 = C4.1 * C10
    probs_q11 = C4_1 * F.softmax(logits_q11, dim=-1)

    # Concatenate probabilities into vector of length 37
    probs = torch.cat([probs_q1, probs_q2, probs_q3, probs_q4, probs_q5, probs_q6,
                       probs_q7, probs_q8, probs_q9, probs_q10, probs_q11], dim=-1)
    return probs
