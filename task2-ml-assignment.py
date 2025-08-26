import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

# converting all the training data into dataframes
print("started creating training data")
train_directory_path = "HAR/Combined/Train"
y_train_list = []
data_frame_train = pd.DataFrame().iloc[0:0]
for dir_name in os.listdir(train_directory_path):
    internal_directory_path_train = os.path.join(train_directory_path, dir_name)
    for file_name in os.listdir(internal_directory_path_train):
        if file_name.endswith(".csv"):
            file_path = os.path.join(internal_directory_path_train, file_name)
            data_frame_train_instance = pd.read_csv(file_path)
            new_list_train = [dir_name] * len(data_frame_train_instance)
            y_train_list.extend(new_list_train)
            data_frame_train = pd.concat([data_frame_train, data_frame_train_instance], ignore_index=True)
data_frame_train['result'] = y_train_list
print("training data size ", data_frame_train.shape)
print("completed creating training data")


# converting all the testing data into dataframes
print("\nstarted creating test data")
test_directory_path = "HAR/Combined/Test"
y_test_list = []
data_frame_test = pd.DataFrame().iloc[0:0]
for dir_name in os.listdir(test_directory_path):
    internal_directory_path_test = os.path.join(test_directory_path, dir_name)
    for file_name in os.listdir(internal_directory_path_test):
        if file_name.endswith(".csv"):
            file_path = os.path.join(internal_directory_path_test, file_name)
            data_frame_test_instance = pd.read_csv(file_path)
            new_list_test = [dir_name] * len(data_frame_test_instance)
            y_test_list.extend(new_list_test)
            data_frame_test = pd.concat([data_frame_test, data_frame_test_instance], ignore_index=True)
data_frame_test['result'] = y_test_list
print("test data size ", data_frame_test.shape)
print("completed creating test data")




def training_decision_tree():
    print("\ndecision tree training started")

    features_list = ['accx', 'accy', 'accz']
    X_train = data_frame_train[features_list]
    y_train = data_frame_train['result']
    X_test = data_frame_test[features_list]
    y_test = data_frame_test['result']

    # Tuning the Hyperparameters with GridSearchCV
    print("\nTuning the Hyperparameters with GridSearchCV  ")
    model = DecisionTreeClassifier(random_state=42)

    # Defining the parameter grid to search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 3, 4, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    # Creating the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fitting the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    print(f"Best parameters found: {grid_search.best_params_}")

    print("decision tree training finished")

    # Evaluating the best model on the test set
    best_dt = grid_search.best_estimator_
    y_pred_tuned = best_dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"Accuracy: {accuracy:.4f}\n")

    precision = precision_score(y_test, y_pred_tuned, average='weighted', zero_division=1)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, y_pred_tuned, average='weighted', zero_division=1)
    print(f"Recall: {recall:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred_tuned)
    print("\nConfusion Matrix:")
    print(conf_matrix)


training_decision_tree()



