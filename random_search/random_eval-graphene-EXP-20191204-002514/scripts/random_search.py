import sys
import os

if __name__ == "__main__":
    n_random_architectures = int(sys.argv[1])
    other_arguments = " ".join(sys.argv[2:])
    cmd_to_run = "python train.py --random " + other_arguments
    for i in range(1, n_random_architectures + 1):
        print("Training random architecture {}/{}...".format(i, n_random_architectures))
        print("    Running {}".format(cmd_to_run))
        os.system(cmd_to_run)
