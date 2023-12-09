import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

f = list(train.columns)
f.remove("Ticket")
f.remove("PassengerId")
f.remove("Embarked")
f.remove("Name")
f.remove("Survived")
f.remove("Fare")
f.remove("Age")
f.remove("Cabin")
y = train["Survived"]


X = pd.get_dummies(train[f])
X_test = pd.get_dummies(test[f])

X.to_csv('x.csv', index=False)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
pred = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
output.to_csv('submission.csv', index=False)


