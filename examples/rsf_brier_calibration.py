# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:15:45 2020
 Example using brier-score and  calibration plot with random survival forrest
@author: Fabian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import brier_score, integrated_brier_score, calibration_curve

if __name__ == '__main__':
    
    X, y = load_gbsg2()
    
    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["'I'", "'II'", "'III'"]]).fit_transform(grade_str)
    
    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt = np.column_stack((Xt.values, grade_num))
    
    feature_names = X_no_grade.columns.tolist() + ["tgrade"]
    
    random_state = 20

    X_train, X_test, y_train, y_test = train_test_split(
    Xt, y, test_size=0.25, random_state=random_state)
    
    rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
    rsf.fit(X_train, y_train)
    
    SurvivalFunction = rsf.predict_survival_function(X_test)
    bs=brier_score(y_train,y_test,SurvivalFunction,rsf.event_times_)