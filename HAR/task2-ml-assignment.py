import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import tsfel
from tqdm import tqdm
import MakeDataset as md

np.random.seed(42)

time = 10
offset = 100
raw_model_accuracy_list = []
feature_model_accuracy_list = []
TSFEL_accuracy_list = []

folders = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

activity_labels_map = {v: k for k, v in classes.items()}
activity_labels = [activity_labels_map[i] for i in sorted(activity_labels_map)]


################ Task 2 Question-1 SubQuestion-1#####################################
def evaluate_model(X_train, y_train, X_test, y_test, model_name="Model",depth = 6, task2_sub_question2 = False):
    print(f"\n--- Evaluating {model_name} ---")
    if task2_sub_question2:
        dt_classifier = DecisionTreeClassifier(max_depth = depth, random_state=42)
    else:
        dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    if not task2_sub_question2:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=activity_labels))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=activity_labels, yticklabels=activity_labels)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}

# The shape is (num_samples, 500, 3). We need to flatten it.
n_samples_train = md.X_train.shape[0]
n_samples_test = md.X_test.shape[0]

X_train_raw_flat = md.X_train.reshape(n_samples_train, -1)
X_test_raw_flat = md.X_test.reshape(n_samples_test, -1)

print(f"Flattened X_train shape: {X_train_raw_flat.shape}")
print(f"Flattened X_test shape: {X_test_raw_flat.shape}")

# Evaluate the model
results_raw = evaluate_model(X_train_raw_flat, md.y_train, X_test_raw_flat, md.y_test, "Raw Data Model (Custom Split)")
for depth in range(1, 9):
    result = evaluate_model(X_train_raw_flat, md.y_train, X_test_raw_flat, md.y_test,
                            "Raw Data Model (Custom Split)", depth, True)
    raw_model_accuracy_list.append(result['Accuracy'])



########################Task 2 Question-1 SubQuestion-2################################3


# TSFEL configuration
cfg = tsfel.get_features_by_domain('statistical')
fs=50
output_dir = 'processed_data'
tsfel_train_path = os.path.join(output_dir, 'X_train_tsfel_custom.csv')
tsfel_test_path = os.path.join(output_dir, 'X_test_tsfel_custom.csv')

def generate_tsfel_features(data_dir, classes_map, tsfel_cfg, sampling_rate):
    all_features_list = []
    labels_list = []
    activity_folders = os.listdir(data_dir)

    for activity_name in tqdm(activity_folders, desc=f"Processing {os.path.basename(data_dir)}"):
        activity_path = os.path.join(data_dir, activity_name)
        if os.path.isdir(activity_path) and activity_name in classes_map:
            label = classes_map[activity_name]
            for file_name in os.listdir(activity_path):
                file_path = os.path.join(activity_path, file_name)
                sample_df = pd.read_csv(file_path)
                features = tsfel.time_series_features_extractor(tsfel_cfg, sample_df, fs=sampling_rate)
                all_features_list.append(features)
                labels_list.append(label)

    X_features = pd.concat(all_features_list, ignore_index=True)
    y_labels = np.array(labels_list)
    X_features.columns = ['_'.join(col).strip() for col in X_features.columns.values]
    X_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_features.fillna(0, inplace=True)
    return X_features, y_labels

if not os.path.exists(tsfel_test_path):
    print("Test features not found. Generating them now...")
    test_data_dir = os.path.join('Combined', 'Test')
    X_test_tsfel, y_test = generate_tsfel_features(test_data_dir, classes, cfg, fs)
    os.makedirs(output_dir, exist_ok=True)
    X_test_tsfel.to_csv(tsfel_test_path, index=False)
    # Also save the corresponding labels for consistency
    np.savetxt(os.path.join(output_dir, 'y_test.txt'), y_test, fmt='%d')
    print(f"Test features saved to {tsfel_test_path}")
else:
    print("Found pre-computed test features.")


def extract_tsfel_features_custom(data_3d, description=""):
    all_features = []
    for i in tqdm(range(len(data_3d)), desc=f"Extracting TSFEL features for {description}"):
        # Create a DataFrame for the current sample (500, 3)
        sample_df = pd.DataFrame(data_3d[i], columns=['accx', 'accy', 'accz'])
        
        # Extract features, sampling frequency (fs) is 50 Hz
        features = tsfel.time_series_features_extractor(cfg, sample_df, fs=50)
        features.columns = ['_'.join(col).strip() for col in features.columns.values]
        all_features.append(features)
        
    return pd.concat(all_features, ignore_index=True)

train_data_dir = os.path.join('Combined', 'Train')
all_features_list = []
labels_list = []

# Get the list of activity folders (e.g., 'WALKING', 'SITTING', etc.)
activity_folders = os.listdir(train_data_dir)

for activity_name in tqdm(activity_folders, desc="Processing Activities"):
    activity_path = os.path.join(train_data_dir, activity_name)

    # Check if it's a directory and if we have a label for it
    if os.path.isdir(activity_path) and activity_name in classes:
        
        # Get the integer label for the current activity
        label = classes[activity_name]
        
        # Loop through each subject's CSV file in the activity folder
        for file_name in os.listdir(activity_path):
            file_path = os.path.join(activity_path, file_name)
            
            # a) Read the CSV file. This is ONE sample.
            # TSFEL expects a pandas DataFrame as input.
            sample_df = pd.read_csv(file_path)
            
            # b) Extract features for this single sample.
            # The output 'features' will be a DataFrame with a SINGLE row.
            features = tsfel.time_series_features_extractor(cfg, sample_df, fs=50)
            
            # c) Append the extracted features (the single row) to our list
            all_features_list.append(features)
            
            # d) Append the corresponding label to our other list
            labels_list.append(label)

# --- 3. FINAL ASSEMBLY ---

print("\nCombining all extracted features...")

# Concatenate the list of single-row DataFrames into one large DataFrame
# This is your final X_train
X_train = pd.concat(all_features_list, ignore_index=True)

# Convert the list of labels into a NumPy array
# This is your final y_train
y_train = np.array(labels_list)

# Clean up the column names that TSFEL creates (e.g., from ('0_Mean', 'accx') to '0_Mean_accx')
X_train.columns = ['_'.join(col).strip() for col in X_train.columns.values]

# Handle any potential NaN/infinite values that TSFEL might produce
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)

# --- 4. VERIFY THE RESULT ---

print("\nProcessing complete!")
print(f"Shape of the final feature matrix X_train: {X_train.shape}")
print(f"Shape of the final labels vector y_train: {y_train.shape}")

# Now you can save this processed data to a file to avoid re-computing
# This is where your 'tsfel_train_path' from the previous question is used
os.makedirs('processed_data', exist_ok=True)
output_path = 'processed_data/X_train_tsfel_custom.csv'
# (Make sure the 'processed_data' directory exists)
X_train.to_csv(output_path, index=False)
print(f"Training features saved to {output_path}")
np.savetxt(os.path.join(output_dir, 'y_train.txt'), y_train, fmt='%d')

print("\nLoading pre-computed TSFEL features and labels...")
y_test = np.loadtxt(os.path.join(output_dir, 'y_test.txt'))
labels_test_path = os.path.join(output_dir, 'y_test.txt')
labels_train_path = os.path.join(output_dir, 'y_train.txt')
# Load the training features you already created
X_train = pd.read_csv(tsfel_train_path)
# We also need the training labels, which we can generate again quickly

y_train = np.loadtxt(labels_train_path)

# Load the testing features
X_test = pd.read_csv(tsfel_test_path)
y_test = np.loadtxt(labels_test_path)
# Load the testing labels


print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")


# --- Step 3: Train the Decision Tree Model ---

print("\nTraining the Decision Tree model...")

# Initialize the classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model using the TSFEL features
dt_classifier.fit(X_train, y_train)

print("Model training complete.")


# --- Step 4: Evaluate the Model and Report Metrics ---

print("\nEvaluating the model on the test set...")

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# --- Report Accuracy, Precision, and Recall ---
accuracy = accuracy_score(y_test, y_pred)
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.4f}\n")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}\n")
for depth in range(1,9):
    dt_classifier = DecisionTreeClassifier(max_depth = depth, random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    TSFEL_accuracy_list.append(accuracy)


print("Classification Report:")
# The classification report provides precision, recall, and F1-score per class
class_names = [name for name, num in sorted(classes.items(), key=lambda item: item[1])]
print(classification_report(y_test, y_pred, target_names=class_names))


# --- Report Confusion Matrix ---
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix for better visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Decision Tree with TSFEL Features')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


############### Task 2 Question-1 SubQuestion-3 ##################

# Path to the original dataset folder
DATASET_PATH = "UCI HAR Dataset/"

# Load the pre-computed features
X_train_provided = pd.read_csv(os.path.join(DATASET_PATH, 'train', 'X_train.txt'), delim_whitespace=True, header=None)
X_test_provided = pd.read_csv(os.path.join(DATASET_PATH, 'test', 'X_test.txt'), delim_whitespace=True, header=None)

# Load the original labels
y_train_provided = pd.read_csv(os.path.join(DATASET_PATH, 'train', 'y_train.txt'), header=None).squeeze()
y_test_provided = pd.read_csv(os.path.join(DATASET_PATH, 'test', 'y_test.txt'), header=None).squeeze()

print(f"\nOriginal Provided X_train shape: {X_train_provided.shape}")
print(f"Original Provided X_test shape: {X_test_provided.shape}")

# Evaluate the model
results_provided = evaluate_model(X_train_provided, y_train_provided, X_test_provided, y_test_provided, "Provided Features Model (Original Split)")
for depth in range(1,9):
    result = evaluate_model(X_train_provided, y_train_provided, X_test_provided, y_test_provided,
                            "Provided Features Model (Original Split)", depth, True)
    feature_model_accuracy_list.append(result['Accuracy'])

#=======================================================================
#================================task 2 Question-2 =======================================


def test_dt_with_various_depths():

    depth_list = range(1, 9, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(depth_list, raw_model_accuracy_list, label='Raw Data Model')
    plt.plot(depth_list, feature_model_accuracy_list, label='Feature Data Model')
    plt.plot(depth_list, TSFEL_accuracy_list, label='TSFEL features Model')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Models Accuracy on various depths')
    plt.legend()
    plt.grid(True)
    plt.show()

test_dt_with_various_depths()