# Pima Indian diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

diabetes_data = pd.read_csv('./datasets/diabetes.csv')
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.head(5))
print(diabetes_data.info())

# train_test_split
# x for feature data, y for label data
x = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=3, stratify=y)
