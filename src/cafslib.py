# -*- coding: utf-8 -*-
"""
iCAFS Library

January - 2025
-- Authors --
Salvador Romo
Miguel De-la-Torre
"""

#import cafslib as cafs
# Load required libraries and functions
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Import the classifiers to be tested
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Libraries to adapt the gaussian naive bayes
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.naive_bayes import GaussianNB - Already loaded
# from sklearn.model_selection import GridSearchCV

# To parallelize
from joblib import Parallel, delayed

####################################################################
# Train the model. By now, there is a single model.
def train_model(model_name):
  """ To train the model
  """
  # m = KNeighborsClassifier(n_neighbors=1)
  
  # Variants of Naïve Bayes
  # m = GaussianNB()
  # m = BernoulliNB() 
  m = ComplementNB(alpha=10.0) 
  # m = MultinomialNB()

  return m

# Original algorithm for covering array feature selection
def cafs(ca, dataset_x, dataset_y, max_iter, model='GaussianNB'):
  """
  CAFS - Stands for Covering Arrays Feature Selection.

  ca - The covering array
  dataset_x - the dataset "data"
  dataset_y - the dataset "labels"
  max_iter - the maximum number of iterations

  model - The classifier to be used in the evaluation
     'GaussianNB'
     'svm'
     'kNeighborsClassifier'

  """

  # Required variables
  global_data_set = dataset_x
  global_max = 0
  max_iteartion = 0
  result_list_x = []
  result_list_y = []
  result_list_score = []
  num_rows = ca.shape[0]

  # Adjust the covering array according to the number of features in the dataset
  if len(dataset_x.columns) < ca.shape[1] :
    num_colums = len(dataset_x.columns)
  else:
    num_colums = ca.shape[1]

  # Iterate over the accuracy
  while max_iteartion < max_iter:

     lst_headers = global_data_set.columns.values.copy()
     max_score = 0.0
     mx_data_set = None
     #random.shuffle(lst_headers)

     for i in range(0,num_rows):
        lst_headers_to_select = []
        for j in range(0,num_colums ):
          if ca[i][j] == 1 :
              lst_headers_to_select.append(lst_headers[j])

        # with the list of headers to select get sub dat set of col with pandas
        if len(lst_headers_to_select) == 0:
            continue
        df_temp = dataset_x[lst_headers_to_select]
        x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(dataset_x[lst_headers_to_select], dataset_y.values.ravel(),test_size=0.20,random_state=42)

        # train a model
        m = train_model( model )
        m.fit(x_train_temp, y_train_temp)
        y_pred = m.predict(x_test_temp)

        # get accuracy in terms of F1-score
        score = f1_score(y_test_temp, y_pred,average='macro')

        # if accuracy is better than maxScore so far assing subDataSet  to local_sub_data_set
        if score >= max_score :
            max_score = score
            mx_data_set = df_temp.copy()

     global_data_set = mx_data_set.copy()
     global_max = max_score
     print(m.get_params())
     print(global_data_set.columns)
     print(global_max)
     print("____________________________")
     num_colums  = len(global_data_set.columns)
     mx_data_set = None
     max_score = 0
     max_iteartion = max_iteartion  +1
     result_list_x.append(max_iteartion)
     result_list_y.append(len(global_data_set.columns))
     result_list_score.append(global_max)

  return result_list_x,result_list_y,result_list_score

