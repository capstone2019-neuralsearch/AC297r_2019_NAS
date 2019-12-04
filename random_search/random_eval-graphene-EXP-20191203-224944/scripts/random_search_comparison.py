import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser("darts")
parser.add_argument('--dataset', type=str, default='cifar', help='name of the dataset to use (e.g. cifar, mnist, graphene)')
args = parser.parse_args()

if __name__ == "__main__":
    random_search_dirs = glob.glob("random_eval-{}-EXP-*".format(args.dataset))

    last_valid_acc = {}
    for exp_dirname in random_search_dirs:
        log_filename = os.path.join(exp_dirname, "log.txt")
        with open(log_filename, "r") as log_file:
            valid_acc_lines = [float(l.strip().split(" ")[-1])
                                   for l in log_file.readlines()
                                   if "valid_acc" in l]
        last_valid_acc[exp_dirname] = valid_acc_lines[-1]
    for exp_dirname, valid_acc in last_valid_acc.items():
        print("{}    valid_acc: {}".format(exp_dirname, valid_acc))
    best_exp_dirname, best_valid_acc = max(last_valid_acc.items(),
                                           key=lambda l: l[1])
    print("Best exp is {} with valid_acc of {}".format(best_exp_dirname,
                                                       best_valid_acc))
