import torch 
from torch import Tensor 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset 
import numpy as np 
from numpy import ndarray 
import scipy 
import scipy.sparse as sp 

import wandb 
import torch_scatter 
import torch_sparse 
import dgl 
import dgl.sparse as dglsp 

import random 
import math 
import os
import sys 
import time 
from datetime import datetime, date  
from dataclasses import dataclass, asdict
from collections import defaultdict, namedtuple, Counter 
import itertools
import functools 
import argparse 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm 
import lzma 
import traceback 
import json 
import csv 
import copy 
import pickle 
from pprint import pprint, pformat
from typing import Optional, Any, Union, Callable, Literal, Iterable, Iterator

pprint = functools.partial(pprint, sort_dicts=False)
pformat = functools.partial(pformat, sort_dicts=False)

IntTensor = FloatTensor = BoolTensor = FloatScalarTensor = SparseTensor = Tensor 
IntArray = FloatArray = BoolArray = ndarray 
FloatArrayTensor = IntArrayTensor = Union[Tensor, ndarray]

INF = int(1e9)

EdgeIndex = IntTensor 
NodeType = str 
EdgeType = tuple[str, str, str]
