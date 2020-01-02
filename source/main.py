import tensorflow as tf
import os
import shutil
from train import training
from valid import validate
from config import config
from test import testing

# Directories
# Validation
valid_intp_dir = config.VALID.intp_dir
logs_valid = config.VALID.logs_valid
# Test
test_dir = config.TEST.datapath
test_intp_dir = config.TEST.intp_dir
logs_test = config.TEST.logs_test
# Model
model_dir = config.TRAIN.model_dir

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='Running process')
  args = parser.parse_args()
  if args.mode == 'train':
    training()
  elif args.mode == 'valid':
    validate(valid_intp_dir, logs_valid, model_dir)
  elif args.mode == 'test':
    testing(test_intp_dir, logs_test, model_dir)
  else:
    raise Exception("Unknown mode")


