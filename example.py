from __future__ import print_function
import numpy as np
import sys
import DI
import scipy
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import matplotlib.pyplot as plt
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='SMK_CAN_187.mat')
args = parser.parse_args()
    
svm_tuned_params = [{'kernel': ['rbf'], 'gamma': [1e0,1e-1,1e-2,1e-3,1e-4], 'C': [1, 10, 100]}]
svc = SVC(kernel='rbf')
    
data = sio.loadmat(args.option)
x_train = data['X']
y_train = data['Y']
print(x_train.shape, y_train.shape)
y_train = y_train[:,0]
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

epsilon = 1e-3; mu = 0.001; num_features = 100; type_Y = 'categorical'
rank_di = DI.di(x_train, y_train, num_features, type_Y, epsilon, mu, learning_rate = 0.1, iterations = 1000, verbose = True)

num = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

print("DI SCORES**********")
res = []
for i in range (20):
    selected_feats_di = np.argsort(rank_di)[:num[i]]
    x_train_selected_di = np.take(x_train, selected_feats_di, axis=1)

    score_di = 0
    for rs in range (5):
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
        clf = GridSearchCV(estimator=svc, param_grid=svm_tuned_params, cv=inner_cv)
        score_di += cross_val_score(clf, X=x_train_selected_di, y=y_train, cv=outer_cv)
    res.append(score_di.mean()/5)
    
plt.plot(num, res, marker='+')
plt.ylabel('KSVM Accuracy')
plt.xlabel('# features selected')
plt.grid(True)
plt.show()

print("full feature**********")
score_full = 0
for rs in range (5):
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
    clf = GridSearchCV(estimator=svc, param_grid=svm_tuned_params, cv=inner_cv)
    score_full += cross_val_score(clf, X=x_train, y=y_train, cv=outer_cv)
print('Full feature accuracy:',score_full.mean()/5)
