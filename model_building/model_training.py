import pandas as pd
import numpy as np
import pickle
import os
from huggingface_hub import HfApi, login, snapshot_download

# library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

# libraries to build Linear Regression Model
from sklearn.linear_model import LogisticRegression

# libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# libraries to build ensemble models
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)

# libraries to build xgboost model
from xgboost import XGBClassifier

# to tune different models
from sklearn.model_selection import GridSearchCV, train_test_split

# to get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
)

# Define metric functions (copied from notebook)
def get_metrics_score(model, X_train, X_test, y_train, y_test, flag=True):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = metrics.accuracy_score(y_train, pred_train)
    test_acc = metrics.accuracy_score(y_test, pred_test)

    train_recall = metrics.recall_score(y_train, pred_train)
    test_recall = metrics.recall_score(y_test, pred_test)

    train_precision = metrics.precision_score(y_train, pred_train)
    test_precision = metrics.precision_score(y_test, pred_test)

    train_f1 = metrics.f1_score(y_train, pred_train)
    test_f1 = metrics.f1_score(y_test, pred_test)

    scores = [train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1]

    if flag:
        print("Accuracy on training set : ", train_acc)
        print("Accuracy on test set : ", test_acc)
        print("Recall on training set : ", train_recall)
        print("Recall on test set : ", test_recall)
        print("Precision on training set : ", train_precision)
        print("Precision on test set : ", test_precision)
        print("F1 on training set : ", train_f1)
        print("F1 on test set : ", test_f1)
    return scores

def draw_matrix(model, predictors, target):
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# Hugging Face login
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
    login(token=HF_TOKEN)
except KeyError:
    print("HF_TOKEN environment variable not set. Please set it.")
    exit()
except Exception as e:
    print(f"Error logging into Hugging Face: {e}")
    exit()

hf_username = os.environ.get('HF_USERNAME', 'your_hf_username_here') # Placeholder for username
processed_dataset_repo_id = f"{hf_username}/tourism-processed-dataset"
model_repo_id = f"{hf_username}/tourism-best-model"

local_data_path = "processed_data"
os.makedirs(local_data_path, exist_ok=True)

# 1. Download processed data
print(f"Downloading processed data from {processed_dataset_repo_id}...")
snapshot_download(repo_id=processed_dataset_repo_id, local_dir=local_data_path, repo_type="dataset")

X_train = pd.read_parquet(os.path.join(local_data_path, "X_train.parquet"))
X_test = pd.read_parquet(os.path.join(local_data_path, "X_test.parquet"))
y_train = pd.read_parquet(os.path.join(local_data_path, "y_train.parquet")).squeeze() # .squeeze() to convert DataFrame back to Series
y_test = pd.read_parquet(os.path.join(local_data_path, "y_test.parquet")).squeeze() # .squeeze() to convert DataFrame back to Series

print("Data loaded successfully.")

# List to store model performance
performance_metrics = []

# Helper function to get and store metrics (modified to accept X_train, X_test, y_train, y_test)
def add_model_metrics(model_name, model_obj):
    scores = get_metrics_score(model_obj, X_train, X_test, y_train, y_test, flag=False)
    performance_metrics.append({
        'Model': model_name,
        'Train Accuracy': scores[0],
        'Test Accuracy': scores[1],
        'Train Recall': scores[2],
        'Test Recall': scores[3],
        'Train Precision': scores[4],
        'Test Precision': scores[5],
        'Train F1-Score': scores[6],
        'Test F1-Score': scores[7]
    })

# 2. Train and Evaluate Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
print("Decision Tree Classifier Performance:")
add_model_metrics('Decision Tree', dtree)

# 3. Train and Evaluate Bagging Classifier
bagging_tree = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                                 random_state=42)
bagging_tree.fit(X_train, y_train)
print("Bagging Classifier Performance:")
add_model_metrics('Bagging Classifier', bagging_tree)

# 4. Train and Evaluate Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("Random Forest Classifier Performance:")
add_model_metrics('Random Forest', rf_classifier)

# 5. Train and Evaluate AdaBoost Classifier
ada_boost = AdaBoostClassifier(random_state=42)
ada_boost.fit(X_train, y_train)
print("AdaBoost Classifier Performance:")
add_model_metrics('AdaBoost', ada_boost)

# 6. Train and Evaluate GradientBoost Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)
print("GradientBoost Classifier Performance:")
add_model_metrics('GradientBoost', gb_classifier)

# 7. Train and Evaluate XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
print("XGBoost Classifier Performance:")
add_model_metrics('XGBoost', xgb_classifier)

# 8. Train and Evaluate Stacking Classifier
estimators = [('dt', dtree),
              ('rf', rf_classifier),
              ('gb', gb_classifier),
              ('xgb', xgb_classifier)]

final_estimator = LogisticRegression(random_state=42)

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    passthrough=True
)

stacking_classifier.fit(X_train, y_train)
print("Stacking Classifier Performance:")
add_model_metrics('Stacking Classifier', stacking_classifier)

# Create DataFrame for performance comparison
performance_df = pd.DataFrame(performance_metrics)
print("\n--- Model Performance Comparison ---")
print(performance_df)

# Identify the best performing model based on 'Test F1-Score'
best_model_row = performance_df.loc[performance_df['Test F1-Score'].idxmax()]
best_model_name = best_model_row['Model']

# Map model names to their actual objects
model_mapping = {
    'Decision Tree': dtree,
    'Bagging Classifier': bagging_tree,
    'Random Forest': rf_classifier,
    'AdaBoost': ada_boost,
    'GradientBoost': gb_classifier,
    'XGBoost': xgb_classifier,
    'Stacking Classifier': stacking_classifier
}

best_model_object = model_mapping[best_model_name]

# Define the path to save the best model
model_save_dir = "tourism_project/model_building"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "best_model.pkl")

# Save the best model using pickle
with open(model_save_path, 'wb') as file:
    pickle.dump(best_model_object, file)

print(f"The best performing model ({best_model_name}) has been saved to {model_save_path}")

# Upload the best model to Hugging Face Hub
api = HfApi()
print(f"Uploading best model to {model_repo_id}...")
api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=model_save_path,
    path_in_repo="best_model.pkl",
    repo_id=model_repo_id,
    repo_type="model",
    commit_message=f"Upload best performing model: {best_model_name}"
)
print(f"Successfully uploaded best model to {model_repo_id}")

print("Model training and registration complete.")
