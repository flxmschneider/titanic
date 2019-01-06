def transform_data(raw_data):
    data = copy(raw_data)
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x=='male' else 0)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    features = ['Age', 'Sex', 'Pclass','Fare', 'SibSp', 'Parch']
    X = data[features]
    if 'Survived' in data.columns.values:
        y = data['Survived']
        ret = (X,y)
    else:
        ret = X
    return ret
