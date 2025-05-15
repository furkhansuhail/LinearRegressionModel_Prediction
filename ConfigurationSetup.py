import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import urllib.request as request
import re
import sys
import statsmodels.api as sm
import math
import missingno as msno # to get visualization on missing values

warnings.filterwarnings('ignore') # To supress warnings
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from dataclasses import dataclass

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth',None)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # To supress numerical display in scientific notations


# set the background for the graphs
from scipy.stats import skew
plt.style.use('ggplot')

print("Load Libraries- Done")


