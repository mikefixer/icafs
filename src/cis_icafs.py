# -*- coding: utf-8 -*-
""" cis_cafs

Script to test the iterative covering array feature selection algorithm using the circle in square synthetic dataset

Salvador Romo and Miguel De-la-Torre
iteso and UdG working together.
"""
# Our Library!!! - still under development
import cafslib as cl

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

imgpath = '../images/'

############################## TEST CASE - CIRCLE IN SQUARE ################################
# INFO: Reading the dataset. Separate the dataframes for features and labels.
dfCIS = pd.read_csv('../data/cis_data.csv')
X = dfCIS[['X', 'Y', 'X10', 'Y10', 'X20', 'Y20', 'X30', 'Y30', 'X40', 'Y40']]
y = dfCIS[['Result']]

# Normalize data (some samples fall outside the feature space, due to additive noise)
scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Remove samples with NaN
X_norm_sNaN = X_norm.dropna(how='any')
y_sNaN = y.loc[X_norm_sNaN.index]

# Feature selection to obtain the selected features and scores
start_time = time.time()
res_scores, res_features, res_iter = cl.icafs(X_norm_sNaN, y_sNaN, 3, 4)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")

# Convert to numpy arrays to operate over them
res_scores = np.array(res_scores)
res_features = np.array(res_features)
res_iter = np.array(res_iter)

# Generate figures
# 1. Features/F1-score and iterations
print('Generating figures...')
fig, ax1 = plt.subplots()
barwidth = 0.4
color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Number of features', color=color)
ax1.set_title("ICAFS Feature selection on the CIS dataset")
ax1.bar(res_iter-0.2, res_features, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(1,max(res_features)+2)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
for i in range(len(res_iter)):
    ax1.text(i+1-0.2,    res_features[i], res_features[i])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('F1_score', color=color)
ax2.bar(res_iter+0.2, res_scores, color=color, width=barwidth)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(min(res_scores)-0.001, max(res_scores)+0.001)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
for i in range(len(res_iter)):
    ax2.text(i+1, res_scores[i], f"{res_scores[i]:.3f}")

plt.savefig(imgpath + 'icafs_cis.png', bbox_inches='tight')
#plt.show()

# 2. Features vs F1-score
fig, ax1 = plt.subplots()
barwidth = 0.4

color = 'tab:blue'
ax1.set_xlabel('Number of features')
ax1.set_ylabel('F1-score', color=color)
ax1.set_title("ICAFS Feature selection on the CIS dataset")
ax1.bar(np.arange(len(res_features))+1, res_scores, color=color, width=barwidth)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(res_features))+1, res_features.astype(str))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_ylim(min(res_scores)-0.001, max(res_scores)+0.001)
for i in range(len(res_features)):
    ax2.text(i+1, res_scores[i], f"{res_scores[i]:.3f}")

plt.savefig(imgpath + 'icafs_cis_scores.png', bbox_inches='tight')
#plt.show()
