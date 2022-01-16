# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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

# %% [markdown]
# ## Kaggle - Red Wine Quality Regression
#
# This notebook contains the regression pipeline for the Red Wine Quality Kaggle dataset. This file is a python module that can be opened as a notebook using jupytext.
#
# 1. [Business/Data Understanding](#1.-Business/Data-Understanding)
# 2. Exploratory Data Analysis
# 3. [Data Preparation](#3.-Data-Preparation)
# 4. Modeling
# 5. Evaluation
#
# [References](#References)

# %%
from pandas import read_csv
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### 1. Business/Data Understanding
#
# Wine quality refers to the factors that go into producing a wine, as well as the indicators or characteristics that tell you if the wine is of high quality.
#
# When you know what influences and signifies wine quality, you’ll be in a better position to make good purchases. You’ll also begin to recognize your preferences and how your favorite wines can change with each harvest. Your appreciation for wines will deepen once you’re familiar with wine quality levels and how wines vary in taste from region to region.
#
# Some wines are higher-quality than others due to the factors described below. From climate to viticulture to winemaking, a myriad of factors make some wines exceptional and others run-of-the-mill.

# %%

# %% [markdown]
# ### 3. Data Preparation
#
# The original dataset is divided into train and validation datasets. The train dataset can be divided again into train and test datasets, other possible aproach is to apply Cross Validation.

# %%
raw_df = read_csv('data/winequality-red.csv', header=0)
train_df, validation_df = train_test_split(
  raw_df,
  test_size=0.2,
  random_state=3
)

train_df.to_csv('data/train.csv', header=True, index=True)
validation_df.to_csv('data/validation.csv', header=True, index=True)

# %% [markdown]
# ### References
#
# [1] [Wine Quality Introduction](https://www.jjbuckley.com/wine-knowledge/blog/the-4-factors-and-4-indicators-of-wine-quality/1009) 
