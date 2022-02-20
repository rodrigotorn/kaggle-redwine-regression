# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="52f39489"
# ## Kaggle - Red Wine Quality Regression
#
# This notebook contains the regression pipeline for the Red Wine Quality Kaggle dataset. This file is a python module that can be opened as a notebook using jupytext.
#
# 1. [Business/Data Understanding](#1.-Business/Data-Understanding)
# 2. [Exploratory Data Analysis](#2.-Exploratory-Data-Analysis)
# 3. [Data Preparation](#3.-Data-Preparation)
# 4. [Modeling](#4.-Modeling)
# 5. [Evaluation](#5.-Evaluation)
#
# [References](#References)

# %% id="2ad3fae2"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings 
warnings.filterwarnings('ignore')
#from google.colab import files

# %% [markdown] id="e5ec0a9e"
# ### 1. Business/Data Understanding
#
# Wine quality refers to the factors that go into producing a wine, as well as the indicators or characteristics that tell you if the wine is of high quality.
#
# When you know what influences and signifies wine quality, you’ll be in a better position to make good purchases. You’ll also begin to recognize your preferences and how your favorite wines can change with each harvest. Your appreciation for wines will deepen once you’re familiar with wine quality levels and how wines vary in taste from region to region.
#
# Some wines are higher-quality than others due to the factors described below. From climate to viticulture to winemaking, a myriad of factors make some wines exceptional and others run-of-the-mill.

# %% [markdown] id="L0e6H1UCeCBy"
# ### 2. Exploratory Data Analysis
#
# The original dataset is divided into train and validation datasets. 
#
# The exploration is applied to the train dataset, since the validation dataset is supposed to be unseen data. The basic aspects of the train data are shown bellow.

# %% id="af169977"
# Run only on the first time
if False:
  raw_df = pd.read_csv('data/winequality-red.csv', header=0)
  train_df, validation_df = train_test_split(
  raw_df,
    test_size=0.2,
    random_state=3
    )

  train_df.to_csv('data/train.csv', header=True, index=True)
  validation_df.to_csv('data/validation.csv', header=True, index=True)

# %% [markdown] id="z0KUfQ0Lgr02"
# From the dataset info, we can see that the data is really clean, there are no null values. The target variable, quality, is composed by integers, while the features are floats.

# %% colab={"base_uri": "https://localhost:8080/"} id="E6eU5u5RfeRK" outputId="edcb346b-5b59-4b3e-c56d-81f9140d7539"
train = pd.read_csv('data/train.csv', header=0, index_col=0)
print(f'Dataset shape: {train.shape} \n')
print(train.info())

# %% [markdown] id="9yAsJjZyhTzw"
# Using the `describe()` method we can see the mean, standard deviation and the quartiles.
#
# It is also worth noting that the quality only has 6 different values.

# %% colab={"base_uri": "https://localhost:8080/", "height": 416} id="0t_gzxmDhUJn" outputId="99003b82-391a-4721-9543-c41513ca1fc1"
print(train['quality'].unique())
train.describe()

# %% [markdown] id="hBzZV6v5kYOc"
# From the correlation plot, on the last row we can see that some features such as **sulphates** and **alcohol** are positive correlated to quality, while others like **volatile acidity** are negative correlated.

# %% colab={"base_uri": "https://localhost:8080/", "height": 638} id="g761yFYmkYoz" outputId="fb885672-d608-46e0-abc8-5391f00bbcb4"
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
  square=True, linewidths=.5, cbar_kws={"shrink": .5})

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="YjgSWgRyliaZ" outputId="293b2cc5-cf80-4619-d55a-10b7c0e6be04"
fig, axes = plt.subplots(12, 1, figsize=(15, 20))
for count, col in enumerate(train.columns):
  sns.distplot(train[col], ax=axes[count])
  axes[count].set_title(col)

# %% [markdown] id="f238f2be"
# ### 3. Data Preparation
#
# Since the data is already very clean, it is only necessary to scale the features. We are going to use 2 different scalers for preprocessing.

# %% id="H7_6yqDJvKwz"
train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1:]

standard_scaler_x = StandardScaler().fit(train_x)
standard_train_x = standard_scaler_x.transform(train_x)
standard_scaler_y = StandardScaler().fit(train_y)
standard_train_y = standard_scaler_y.transform(train_y)

minmax_scaler_x = MinMaxScaler().fit(standard_train_x)
scaled_train_x = minmax_scaler_x.transform(standard_train_x)
minmax_scaler_y = MinMaxScaler().fit(standard_train_y)
scaled_train_y = minmax_scaler_y.transform(standard_train_y)
scaled_train_y = scaled_train_y.astype('str')

# %% [markdown] id="bHUePLO70vPb"
# ### 4. Modeling

# %% id="BAHEKcMu0vvk" tags=[]
# Run only for grid search
if False:
  params = {
    'hidden_layer_sizes': [(i, ) for i in range(1, 22, 2)],
    'activation': ('identity', 'logistic', 'tanh', 'relu'),
    'batch_size': [50, 100, 200],
    'max_iter': [200, 400, 800],
    'alpha': [0.0001, 0.01, 1],
    }

  base_mlp = MLPClassifier(random_state=3)
  mlp = GridSearchCV(
    base_mlp,
    params,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    cv=5,
    return_train_score=True,
  )
  mlp.fit(scaled_train_x, scaled_train_y)

  results = pd.DataFrame(mlp.cv_results_)
  results = results[['param_hidden_layer_sizes', 'param_activation',
    'param_batch_size', 'param_max_iter', 'param_alpha',
    'mean_test_score', 'rank_test_score','mean_train_score']]
  results.to_csv('mlp_results.csv', index=False)
  # files.download('mlp_results.csv')

# %% [markdown] tags=[]
# The best results are clealy overfitted. To adjust this ranking, one column were added, which represents the distance between test and train errors. Rows which the gap was greater than 5% were dropped from the ranking.

# %%
results = pd.read_csv('outputs/mlp_results.csv')
results['test/train'] = \
  results['mean_test_score']/results['mean_train_score']
results[results['test/train'] \
  < 1.05].sort_values(['rank_test_score'])

# %%
sample = results.query(
  """param_activation == 'relu' \
  & param_batch_size == 50 \
  & param_max_iter == 800 \
  & param_alpha == 0.0001"""
)
plt.plot(
  sample['param_hidden_layer_sizes'],
  sample['mean_train_score'],
  '-o', label='Train')
plt.plot(
  sample['param_hidden_layer_sizes'],
  sample['mean_test_score'],
  '-o', label='Test')
plt.legend()
plt.title('Overfit Analysis')
plt.show()

# %% [markdown]
# ### 5. Evaluation

# %%
validation = pd.read_csv('data/validation.csv',
  header=0, index_col=0)

validation_x = validation.iloc[:,:-1]
validation_y = validation.iloc[:,-1:]
validation_y.reset_index(drop=True, inplace=True)

standard_validation_x = standard_scaler_x.transform(validation_x)
scaled_validation_x = minmax_scaler_x.transform(standard_validation_x)

# %% tags=[]
mlp = MLPClassifier(
  hidden_layer_sizes=(21,),
  activation='relu',
  batch_size=50,
  max_iter=800,
  alpha=0.0001,
  random_state=3
)
mlp.fit(scaled_train_x, scaled_train_y)
results = standard_scaler_y.inverse_transform(
  minmax_scaler_y.inverse_transform(
    mlp.predict(scaled_validation_x).reshape((320,1))
  )
)

# %%
validation_y['pred'] = pd.Series(results.astype('int').reshape((320,)))
validation_y.to_csv('outputs/predictions.csv', index=False)

print(accuracy_score(
  validation_y['quality'],
  validation_y['pred'])
)

# %% [markdown] id="d4c7df44"
# ### References
#
# [1] [Wine Quality Introduction](https://www.jjbuckley.com/wine-knowledge/blog/the-4-factors-and-4-indicators-of-wine-quality/1009) 
