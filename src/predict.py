import pickle
import pandas as pd
from transformation import transform_data

filename = '../models/gridsearch_RandomForest_1.sav'#2019-01-13 19:11_0.8197424892703863.sav'
best_grid = pickle.load(file=open(filename, 'rb'))

test_set = pd.read_csv('../data/test.csv')
X = transform_data(test_set)

pass_id = test_set['PassengerId'].values
res = best_grid.predict(X)

filename = 'submission.csv'
f = open(filename, 'w+')
f.write('PassengerId,Survived \n')
for p, r in zip(pass_id, res):
    f.write(str(p)+','+str(r)+'\n')
f.close()
print('Prediction written to '+str(filename))