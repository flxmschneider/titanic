import copy
import pandas as pd
import numpy as np

def transform_data(raw_data):
        data = copy.copy(raw_data)
        features = ['Age', 'Sex', 'Pclass','Fare', 'SibSp', 'Parch', 'Embarked','Survived']
                
        data.loc[:,'Sex'] = data['Sex'].apply(lambda x: 1 if x=='male' else 0)
        data.loc[:,'Embarked'] = data['Embarked'].replace({'S':int(1),'C':2,'Q':3})
        data.loc[:,'Age'] = pd.cut(data['Age'],bins=np.arange(0,90,5),labels=np.arange(1,18))
        data.loc[:,'Fare'] = pd.cut(data['Fare'],bins=np.arange(0,700,100),labels=np.arange(1,7))
        data.fillna(1, inplace=True)
        X = data[features[:7]]
        if 'Survived' in data.columns.values:
            y = data['Survived']
            ret = (X,y)
        else:
            ret = X
            return ret
