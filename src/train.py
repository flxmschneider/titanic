import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import pickle
import copy

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformation import transform_data

raw_data = pd.read_csv('../data/train.csv')
data = copy.copy(raw_data)

X,y = transform_data(raw_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

param_grid = {
    'bootstrap': [True],
    'max_depth': [75, 80, 85],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [2, 3, 4 ],
    'min_samples_split': [7, 8, 9],
    'n_estimators': [190, 200, 210]
}
grid_search = False

if grid_search == True:
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    
    best_grid = grid_search.best_estimator_
    best_grid.fit(X_train,y_train)
    pred = best_grid.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print('Accuracy score: ',acc)
    filename= '../models/gridsearch_RandomForest_'+str(pd.Timestamp.now())[:16]+'_'+str(acc)+'.sav'
    pickle.dump(best_grid, open(filename, 'wb'))

else:
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=75, max_features=2, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=4, min_samples_split=7,
                min_weight_fraction_leaf=0.0, n_estimators=210, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print('Accuracy: {}'.format(acc))
    filename= '../models/gridsearch_RandomForest_'+str(pd.Timestamp.now())[:16]+'_'+str(acc)+'.sav'
    model.fit(X, y)
    pickle.dump(model, open(filename, 'wb'))  
    
