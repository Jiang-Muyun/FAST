"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
Currently under review for publication.
Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
torch.set_num_threads(6)
import h5py
import torch.nn as nn

import os
import time
import json
import datetime
import string
import random
import numpy as np
from itertools import chain

def bold(x):       return '\033[1m'  + str(x) + '\033[0m'
def dim(x):        return '\033[2m'  + str(x) + '\033[0m'
def italicized(x): return '\033[3m'  + str(x) + '\033[0m'
def underline(x):  return '\033[4m'  + str(x) + '\033[0m'
def blink(x):      return '\033[5m'  + str(x) + '\033[0m'
def inverse(x):    return '\033[7m'  + str(x) + '\033[0m'
def gray(x):       return '\033[90m' + str(x) + '\033[0m'
def red(x):        return '\033[91m' + str(x) + '\033[0m'
def green(x):      return '\033[92m' + str(x) + '\033[0m'
def yellow(x):     return '\033[93m' + str(x) + '\033[0m'
def blue(x):       return '\033[94m' + str(x) + '\033[0m'
def magenta(x):    return '\033[95m' + str(x) + '\033[0m'
def cyan(x):       return '\033[96m' + str(x) + '\033[0m'
def white(x):      return '\033[97m' + str(x) + '\033[0m'

def convert_to_number(value):
    try:
        return int(value) if value.isdigit() else float(value)
    except ValueError:
        return value

def find_available_path(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            return folder
    raise FileNotFoundError('None of the given path exists' + str(folder_list))

def now(fmt = "%Y-%m-%d_%H:%M:%S"):
    return str(datetime.datetime.now().strftime(fmt))

def random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class Tick():
    def __init__(self, name='', silent=False):
        self.name = name
        self.silent = silent

    def __enter__(self):
        self.t_start = time.time()
        if not self.silent:
            print('%s ' % (self.name), end='', flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta

        if not self.silent:
            print(yellow('[%.3fs]' % (self.delta), ), flush=True)

class Tock():
    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name+':'
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta
        if self.report_time:
            print(yellow(self.name)+cyan('%.3fs'%(self.delta)), end=' ', flush=True)
        else:
            print(yellow('.'), end='', flush=True)
