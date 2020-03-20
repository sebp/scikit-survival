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
    
    rsf = RandomSurvivalForest(n_estimators=5000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
    rsf.fit(X_train, y_train)
    
    label='Random Surival Forrest'
    fu_time=365.25 * 3
    # predict SurvivalFunction from rsf model
    SurvivalFunction = rsf.predict_survival_function(X_test)
    times = rsf.event_times_
    #calculate Brier Scores at fu_time = 3 years

    ti,bs=brier_score(y_train,y_test,SurvivalFunction,rsf.event_times_,t_max=fu_time)
    ibs=integrated_brier_score(y_train,y_test,SurvivalFunction,rsf.event_times_,t_max=fu_time)
    print("Performance of Model <%s> ; at %s years follow-up" % (label,'{0:.1f}'.format(fu_time/365.25)))
    print("Brier Score: %s" %  '{0:.4f}'.format(bs[-1]))
    print("Integrated Brier Score: %s" % '{0:.4f}'.format(ibs))
    # calc overal ibs with t_max=None
    ibs=integrated_brier_score(y_train,y_test,SurvivalFunction,rsf.event_times_,t_max=None)
    print("Overall Integrated Brier Score: %s" % ('{0:.4f}'.format(ibs)))
    # plot calibration curve based on traditional binned approach
    (globmin_x,globmax_x,globmin_y, globmax_y) = calibration_curve(y_train,y_test,SurvivalFunction,times,fu_time=fu_time,internal_validation=True,pseudovals=False)
    # plot high-res calibration curve based on jackknife pseudovalues and loess smoothing
    (globmin_x,globmax_x,globmin_y, globmax_y) = calibration_curve(y_train,y_test,SurvivalFunction,times,fu_time=fu_time,internal_validation=True)
    globmin=min(globmin_x,globmin_y)
    globmax=max(globmax_x,globmax_y)
    plt.xlim([globmin,globmax])
    plt.ylim([globmin,globmax])
    plt.plot([globmin,globmax],[globmin,globmax],'--', c='red')
    plt.show()
    