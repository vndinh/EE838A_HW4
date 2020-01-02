import tensorflow as tf
import numpy as np
import time
import os
import random
import imageio

from config import config

def write_logs(filename, log, start=False):
  print(log)
  if start == True:
    f = open(filename, 'w')
    f.write(log + '\n')
  else:
    f = open(filename, 'a')
    f.write(log + '\n')
    f.close()

def img_read(img_dir):
  img = imageio.imread(img_dir, 'PNG-FI')
  h, w, _ = img.shape
  return img, h, w
 
def img_write(img_dir, img, fmt):
  _, h, w, c = img.shape
  img = np.clip(img, 0, 255)
  img = img.astype('uint8')
  img = np.reshape(img, [h,w,c])
  imageio.imwrite(img_dir, img, fmt)

