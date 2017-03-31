import numpy as np
import pandas as pd
from my_settings import (tf_folder, subjects_select)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ctl_left = []
ctl_right = []
ent_left = []
ent_right = []

for subject in subjects_select:
    data = np.load(tf_folder + "%s_ali.npy" % subject)
    ctl_left.append(data[0, :])
    ctl_right.append(data[1, :])
    ent_left.append(data[2, :])
    ent_right.append(data[3, :])

ctl_left = np.asarray(ctl_left)
ctl_right = np.asarray(ctl_right)
ent_left = np.asarray(ent_left)
ent_right = np.asarray(ent_right)

X = np.vstack((ctl_left, ctl_right, ent_left, ent_right))
y = np.concatenate((np.zeros(len(ctl_left)), np.ones(len(ctl_right)), np.ones(
    len(ent_left) * 2), np.ones(len(ent_right) * 3)))

cv = StratifiedShuffleSplit(y, n_iter=10)
ada_params = {
    "adaboostclassifier__n_estimators": np.arange(1, 50, 1),
    "adaboostclassifier__learning_rate": np.arange(0.01, 1, 0.1)
}

ada = AdaBoostClassifier
scaler_pipe = make_pipeline(StandardScaler(), AdaBoostClassifier())
grid = GridSearchCV(scaler_pipe, param_grid=ada_params, cv=cv)

grid.fit(X, y)

ada = grid.best_estimator_

scores = cross_val_score(ada, X, y, cv=cv, scoring="accuracy")
