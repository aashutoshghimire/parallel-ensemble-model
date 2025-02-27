import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
data=pd.read_csv("creditcard.csv",sep=",")

# Prepare the features and target
X = data.drop(['class'], axis=1)

y = (data['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Function to train a single decision tree
def train_tree(X_train, y_train, random_state):
    tree = RandomForestClassifier(n_estimators=1, random_state=random_state)
    tree.fit(X_train, y_train)
    return tree

# # Generate a synthetic dataset for classification
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Number of trees in the forest
n_estimators = 100

cores = [1,2,4,8,10,12,14,16]

for core in cores:

    # Track the training time
    start_time = time.time()
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=core) as executor:
        # Train multiple trees in parallel
        futures = [executor.submit(train_tree, X_train, y_train, i) for i in range(n_estimators)]

        # Collect the trained trees
        trees = [future.result() for future in futures]

    end_time = time.time()
    training_time = end_time - start_time
#     print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training Time: for {core} cores is {training_time:.2f} seconds")

    # Function to aggregate predictions from all trees
    def predict_ensemble(trees, X):
        predictions = np.zeros((X.shape[0], len(trees)), dtype=int)
        for i, tree in enumerate(trees):
            predictions[:, i] = tree.predict(X)
        # Use majority vote for final prediction
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)



    # Make predictions on the test data
    y_pred = predict_ensemble(trees, X_test)




    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy For Random Forest: {accuracy * 100:.2f}%")