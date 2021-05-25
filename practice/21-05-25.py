import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
from operator import truediv
import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dt_clf = DecisionTreeClassifier(random_state=3)
iris_data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.2, random_state=10)

dt_clf.fit(x_train, y_train)


export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names,
                feature_names=iris_data.feature_names, impurity=True, filled=True,
                )

feature_importance = np.round(dt_clf.feature_importances_, 3)
sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)
print(feature_importance)
plt.show()
