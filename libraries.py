from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from pandas.io.parsers import read_csv
from scipy.interpolate import interp1d
import random

# Import datasets, classifiers and performance metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from os import chdir, getcwd
print('done')