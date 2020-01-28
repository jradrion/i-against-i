from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow import set_random_seed
from cleverhans.attacks import FastGradientMethod
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module,unused-import
from tensorflow.python.platform import app, flags # pylint: disable=no-name-in-module,unused-import
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans_tutorials import check_installation
check_installation(__file__)

import glob
import pickle
import sys
import msprime as msp
import numpy as np
import os
import multiprocessing as mp
import shlex
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

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model,Sequential,model_from_json
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D,concatenate, MaxPooling1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

