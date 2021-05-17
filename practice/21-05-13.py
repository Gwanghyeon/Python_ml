'''
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print()
titanic_df = pd.read_csv('./datasets/titanic_train.csv')

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)


titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].value_counts())
print(titanic_df['Cabin'].head(3))

print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())


def get_category(age):
    cate = ''
    if age <= -1:
        cate = 'Unknown'
    elif age <= 5:
        cate = 'Baby'
    elif age <= 12:
        cate = 'Child'
    elif age <= 18:
        cate = 'Teenager'
    elif age <= 25:
        cate = 'Student'
    else:
        cate = 'Elderly'

    return cate


group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Elderly', 'Unknown']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex',
            data=titanic_df, order=group_names)
plt.show()


def encode_features(data_df):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        l_encoder = LabelEncoder()
        l_encoder.fit(data_df[feature])
        data_df[feature] = l_encoder.transform(data_df[feature])

    return data_df


titanic_df = encode_features(titanic_df)
print(titanic_df.head(5))
'''

# Summarizing the contents
# Predicting the possibility of surviving up to the features

# Fuctions
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib.pyplot import axis
from sklearn.preprocessing import LabelEncoder
import numpy as np


def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna('N', inplace=True)
    return df


def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df


def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


def transform_df(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# Data Loading
titanic_df = pd.read_csv('./datasets/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis=1)
x_titanic_df = transform_df(x_titanic_df)


x_train, x_test, y_train, y_test = train_test_split(
    x_titanic_df, y_titanic_df, test_size=0.2, random_state=7)

#DecisionTreeClassifier, RandomForest, LogisticRegression

dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()

dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_precition = round(accuracy_score(y_test, dt_pred), 4)

rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
rf_prediction = round(accuracy_score(y_test, rf_pred), 4)

lr_clf.fit(x_train, y_train)
lr_pred = lr_clf.predict(x_test)
lr_prediction = round(accuracy_score(y_test, lr_pred), 4)

print(dt_precition, rf_prediction, lr_prediction)


def exec_kfold(clf, folds=5):
    kfold = KFold(n_splits=folds)
    score = []

    for train_index, test_index in kfold.split(x_titanic_df):
        x_train, x_test = x_titanic_df.values[train_index], x_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        score.append(accuracy)

    print(score)
    print("mean score: ", np.mean(score))


exec_kfold(dt_clf)

scores = cross_val_score(dt_clf, x_titanic_df, y_titanic_df, cv=5)
for accuracy in scores:
    print(round(accuracy, 4))

print("mean of accuracy(cross_val): ", round(np.mean(scores), 5))
