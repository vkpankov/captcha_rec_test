import os
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser(
    description="Split dataset to train and test folders"
)

parser.add_argument(
    "--source",
    metavar="source",
    default="dataset/all",
    type=str,
    help="Source directory with images",
)
parser.add_argument(
    "--dest_train",
    metavar="device",
    default="dataset/train",
    type=str,
    help="Train data",
)

parser.add_argument(
    "--dest_test",
    metavar="dest_test",
    default="dataset/test",
    type=str,
    help="Test data",
)

parser.add_argument(
    "--test_ratio",
    metavar="epochs_count",
    default=0.2,
    type=float,
    help="Split test ratio",
)
args = parser.parse_args()

file_names = np.array(os.listdir(args.source))
np.random.shuffle(file_names)
train_files, test_files = np.split(
    file_names, [int(len(file_names) * (1 - args.test_ratio))]
)
train_files = [args.source + "/" + name for name in train_files.tolist()]
test_files = [args.source + "/" + name for name in test_files.tolist()]

for name in train_files:
    shutil.copy(name, args.dest_train + "/" + os.path.basename(name))
for name in test_files:
    shutil.copy(name, args.dest_test + "/" + os.path.basename(name))
