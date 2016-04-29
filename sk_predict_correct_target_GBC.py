import numpy as np
from my_settings import *
import mne

from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


epochs = mne.read_epochs(epochs_folder + "0005_target-epo.fif",
                         preload=False)
times = epochs.times

from_time = np.abs(times + 0.9).argmin()
to_time = np.abs(times - 0.1).argmin()


sides = ["left", "right"]
conditions = ["ctl", "ent"]
rois = ["lh", "rh"]
phase = ["in_phase", "out_phase"]
correct = ["correct", "incorrect"]


# data = np.empty([30, len(times[from_time:to_time])])
X_data =[]

for c in correct:
    for subject in subjects_select:
        for condition in conditions:
            for side in sides:
                for roi in rois:
                    for p  in phase:
                        data = np.load(tf_folder +
                                       "%s_pow_%s_%s_MNE_%s_%s_Brodmann.17-%s_target.npy" %
                                       (subject,
                                        condition,
                                        side,
                                        c,
                                        p,
                                        roi))
                        X_data.append(data[:, :, from_time:to_time].mean(axis=0).mean(axis=0))




y = data["corr"].get_values()
X = data_dv.get_values()


cv = StratifiedShuffleSplit(y, n_iter=10)
ada_params = {"gradientboostingclassifier__n_estimators": np.arange(1, 50, 1),
              "gradientboostingclassifier__learning_rate":
              np.arange(0.01, 1, 0.2),
              "gradientboostingclassifier__max_depth": np.arange(1, 8, 2)}

scaler_pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier())
grid = GridSearchCV(scaler_pipe, param_grid=ada_params, cv=cv)

grid.fit(X, y)

ada = ada_grid.best_estimator_

scores = cross_val_score(ada, X, y, cv=cv, scoring="roc_auc")
