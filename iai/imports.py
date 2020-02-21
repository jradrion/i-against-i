from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import pickle
import sys
import math
import msprime as msp
import numpy as np
import os
import multiprocessing as mp
import shutil
import random
import copy
import argparse
import h5py
import allel
import time

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from absl import app, flags
from easydict import EasyDict

from cleverhans.future.tf2.attacks import projected_gradient_descent, fast_gradient_method, spsa

