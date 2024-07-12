import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

from sklearn.ensemble import BaggingClassifier


data=pd.read_csv("creditcard.csv",sep=",")

# Prepare the features and target
X = data.drop(['class'], axis=1)

y = (data['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Function to train a single Bagging classifier
def train_bagging(X_train, y_train, random_state):
    bagging = BaggingClassifier(n_estimators=1, random_state=random_state)
    bagging.fit(X_train, y_train)
    return bagging



# Number of classifiers in the ensemble
n_estimators = 100


cores = [1,2,4,8,10,12,14,16]

for core in cores:

    # Track the training time
    start_time = time.time()

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Train multiple classifiers in parallel
        futures = [executor.submit(train_bagging, X_train, y_train, i) for i in range(n_estimators)]

        # Collect the trained classifiers
        classifiers = [future.result() for future in futures]

    end_time = time.time()
    training_time = end_time - start_time
#     print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training Time: for {core} cores is {training_time:.2f} seconds")

    # Function to aggregate predictions from all classifiers
    def predict_ensemble(classifiers, X):
        predictions = np.zeros((X.shape[0], len(classifiers)), dtype=int)
        for i, classifier in enumerate(classifiers):
            predictions[:, i] = classifier.predict(X)
        # Use majority vote for final prediction
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

    # Make predictions on the test data
    y_pred = predict_ensemble(classifiers, X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for Bagging Classifier: {accuracy * 100:.2f}%")
