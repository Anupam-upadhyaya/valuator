from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
import pandas as pd


 
data = pd.read_csv('final1.csv')
X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y = data.iloc[:, 13].values



# X = data.drop('price_details',axis=1)
# X = X.drop_duplicates()


# print(data.columns)
# data = datasets.load_breast_cancer()

# X = data.drop['PredictedPrice', axis =1]
# y = data['price_details']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=10)
clf.fit(X, y)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(predictions)

print(data)
print(acc)