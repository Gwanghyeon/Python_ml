from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from os import sep
import pandas as pd
import matplotlib.pyplot as plt

'''
feature_name_df = pd.read_csv('./datasets/human_activity/features.txt',
                              sep='\s+', header=None, names=['column_index', 'column_names'])
feature_name = feature_name_df.iloc[:, 1].values.tolist()

# 중복된 피쳐명의 수정
feature_dup_df = feature_name_df.groupby('column_names').count()
print(feature_dup_df)
feature_dup_names_list = feature_dup_df[feature_dup_df['column_index'] > 1]
print(feature_dup_names_list.count())
'''
# 중복 피쳐명 수정 함수
'''

def get_applied_names(old_feature_names):
    feature_dup_df = pd.DataFrame(data=old_feature_names.groupby(
        'column_name').cumcount(), columns=['dup_cnt'])
    print(feature_dup_df)
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_dup_df = pd.merge(
        old_feature_names.reset_index(), feature_dup_df, how='outer')
    new_feature_dup_df['column_names'] = new_feature_dup_df[['column_name', 'dup_cnt']].apply(
        lambda x: x[0]+'_'+str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_dup_df = new_feature_dup_df.drop(['index'], axis=1)
    return new_feature_dup_df


def get_human_dataset():
    feature_name_df = pd.read_csv('./datasets/human_activity/features.txt',
                                  sep='\s+', header=None, names=['column_index', 'column_name'])
    new_feature_name_df = get_applied_names(feature_name_df)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    x_train = pd.read_csv(
        './datasets/human_activity/train/X_train.txt', sep='\s+', names=feature_name)
    x_test = pd.read_csv(
        './datasets/human_activity/test/X-test.txt', sep='\s+', names=feature_name)

    y_train = pd.read_csv('./datasets/human_activity/train/y_train.txt',
                          sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('./datasets/human_activity/test/y_test.txt',
                         sep='\s+', header=None, names=['action'])

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_human_dataset()
print(x_train.info())
'''


def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby(
        'column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    print(feature_dup_df[feature_dup_df['dup_cnt'] > 0])
    new_feature_name_df = pd.merge(
        old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(
        lambda x: x[0]+'_'+str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df


def get_human_dataset():
    feature_name_df = pd.read_csv('./datasets/human_activity/features.txt',
                                  sep='\s+', header=None, names=['column_index', 'column_name'])
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    x_train = pd.read_csv(
        './datasets/human_activity/train/X_train.txt', sep='\s+', names=feature_name)
    x_test = pd.read_csv(
        './datasets/human_activity/test/X_test.txt', sep='\s+', names=feature_name)

    y_train = pd.read_csv('./datasets/human_activity/train/y_train.txt',
                          sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('./datasets/human_activity/test/y_test.txt',
                         sep='\s+', header=None, names=['action'])

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_human_dataset()
print(x_train.info())
print(x_train.head(5))
print(y_train['action'].value_counts())


dt_clf = DecisionTreeClassifier(random_state=7)
dt_clf.fit(x_train, y_train)
pred = dt_clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print('Accuracy: {0:.4f}'.format(accuracy))
print('hyper parameters: ', dt_clf.get_params())


params = {'max_depth': [6, 8, 10, 12, 16, 18, 20]}

grid_cv = GridSearchCV(dt_clf, param_grid=params,
                       scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(x_train, y_train)
print('Accuracy of Gridcv: {0:.4f}'.format(grid_cv.best_score_))
print('Best parameter: ', grid_cv.best_params_)

cv_result_df = pd.DataFrame(grid_cv.cv_results_)

print(cv_result_df[['param_max_depth', 'mean_test_score']])
