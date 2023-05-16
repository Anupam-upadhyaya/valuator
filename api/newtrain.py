import pickle
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



# rf = RandomForestClassifier(n_estimators=100, max_depth=5)
df = pd.read_csv("final1.csv")
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y = df.iloc[:, 13].values

# rf.fit(X, y)

print(X)
print(y)
 # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#  X, y, test_size=0.1, random_state=1234
#  )
# clf = RandomForest(20)
# clf.fit(X, y)
# with open("random_forest_model2.pkl", "wb") as f:
#         pickle.dump(clf, f)


# with open("random_forest_model2.pkl", "rb") as f:
#         clf = pickle.load(f)

# new_X = np.array([[4,8000,1,1,6,64,0.4,1,0,0,1,0,1]])        







# def train_random_forest(n_trees):

#     df = pd.read_csv("final1.csv")
#     # print(df)
#     y = df['PredictedPrice']

#     X = df.drop('PredictedPrice', axis=1)

    # Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1234
    )

    # Train a random forest classifier with the specified number of trees
# clf = RandomForest(n_trees=n_trees)
# clf.fit(X_train, y_train)

#     # Save the trained model to a file
#     with open("random_forest_model2.pkl", "wb") as f:
#         pickle.dump(clf, f)
def predict_random_forest(X):
    # Load the saved model from the file
    with open("random_forest_model2.pkl", "rb") as f:
        clf = pickle.load(f)

    # Make predictions using the trained model
    predictions = clf.predict(X)

    # Return the predictions
    return predictions


# Train the random forest classifier with 20 trees and save it to a file
# train_random_forest(20)

# Load some data to make predictions on
# new_X = np.array(['Labrador Retriever',12,24.5,24.5,80,70,22.5,22.5,65,55,5,5,4,2,2,1,5,5,3,5,5,3])
new_X = np.array([[4,80000,1,1,6,64,0.4,1,0,0,1,0,1]])   
predictions = predict_random_forest(new_X)

# # Make predictions on the new data using the saved modelpredictions = predict_random_forest(X_test)

# Print the predictions
print(predictions)
 