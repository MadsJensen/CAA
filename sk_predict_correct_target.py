import numpy as np
import pandas as pd
from my_settings import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv(data_path +
                   "alpha_mean_pow_data_extracted_phase_target.csv")
data = data.drop("mean", 1)
data["corr"], corr_lbl = pd.factorize(data.correct)

data_dv = pd.get_dummies(data[["ROI", "condition_side",
                               "condition_type", "phase"]])
data_dv["pow"] = data.power

data_itc = pd.read_csv(data_path +
                       "alpha_mean_itc_data_extracted_phase_target.csv")
data_itc = data_itc.drop("mean", 1)

data_dv["itc"] = data_itc["itc"]

y = data["corr"].get_values()
X = data_dv.get_values()


cv = StratifiedShuffleSplit(y, n_iter=10)
ada_params = {"adaboostclassifier__n_estimators": np.arange(1, 50, 1),
              "adaboostclassifier__learning_rate": np.arange(0.01, 1, 0.1)}

ada = AdaBoostClassifier
scaler_pipe = make_pipeline(StandardScaler(), AdaBoostClassifier())
grid = GridSearchCV(scaler_pipe, param_grid=ada_params, cv=cv)

grid.fit(X, y)

ada = grid.best_estimator_

scores = cross_val_score(ada, X, y, cv=cv, scoring="roc_auc")
