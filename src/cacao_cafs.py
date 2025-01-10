# -*- coding: utf-8 -*-
"""CAFS_ex03

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""

# Our Library!!! - still under development
import cafslib as cl

# Load required libraries and functions
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import math

#from testflows.combinatorics import Covering


imgpath = '../images/'

# Load data
covering_array  = np.loadtxt("../data/coveringArray.csv", delimiter=",", dtype=int)

############################### TEST CASE - CACAO #####################################
df = pd.read_csv('../data/cacao.csv')
df_x = df.iloc[:, 1:]
df_y = df.iloc[:, 0:1]

scaler = MinMaxScaler()
df_norm_cacao = pd.DataFrame(scaler.fit_transform(df_x), columns=df_x.columns)

# Remove samples with NaN
df_norm_cacao_sinNaN = df_norm_cacao.dropna(how='any')
df_y_clean = df_y.loc[df_norm_cacao_sinNaN.index]



res_x,res_y,res_score = cl.cafs(covering_array, df_norm_cacao_sinNaN, df_y_clean, 10)

# Generate the figures
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Number of Features Selected for Cacao Nibs')
ax1.set_title("Features/Iteration")
ax1.bar(res_x, res_y, color = "red")

ax2.set_title("Scores/Iteration")
ax2.bar(res_x, res_score, color = "red")
ax2.set_ylim([0.5, 1.0])

plt.savefig(imgpath + 'cacao_cafs.png', bbox_inches='tight')

