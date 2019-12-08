import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, num_channels=3):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._num_channels = num_channels

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, num_channels=3):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._num_channels = num_channels

    self.stem0 = nn.Sequential(
      nn.Conv2d(self._num_channels, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
        
class NetworkGalaxyZoo(nn.Module):

  def __init__(self, C, num_classes, layers, genotype, fc1_size: int, fc2_size: int, num_channels=3):
    super(NetworkGalaxyZoo, self).__init__()
    self._layers = layers
    self._num_channels = num_channels

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)

    # Fully connected layers
    self.num_fc_layers = 2 if fc2_size > 0 else 1
    self.fc1 = nn.Linear(C_prev, fc1_size)
    # self.fc1 = Maxout(C_prev, fc1_size, 2)
    if self.num_fc_layers >= 2:
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # self.fc2 = Maxout(fc1_size, fc2_size, 2)
    fc_last_size = fc2_size if self.num_fc_layers == 2 else fc1_size

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

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    out = self.global_pooling(s1)

    # Fully connected layers
    conv_out = out.view(out.size(0),-1)
    fc1_out = self.fc1(conv_out)
    if self.num_fc_layers >= 2:
        fc2_out = self.fc2(fc1_out)
        fc_out = fc2_out
    else:
        fc_out = fc1_out
        
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

    # No auxiliary outputs for this model; include for API consistency
    probs_aux = None
    return probs, probs_aux

