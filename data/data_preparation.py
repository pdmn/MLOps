import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login, snapshot_download
import os

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

hf_username = "<YOUR_HF_USERNAME>" # Placeholder for username
raw_dataset_repo_id = f"{hf_username}/tourism-raw-dataset"
processed_dataset_repo_id = f"{hf_username}/tourism-processed-dataset"
local_data_path = "data_raw"

# 1. Download raw dataset
print(f"Downloading raw dataset from {raw_dataset_repo_id}...")
snapshot_download(repo_id=raw_dataset_repo_id, local_dir=local_data_path, repo_type="dataset")
df = pd.read_csv(os.path.join(local_data_path, "tourism.csv"))

# --- Preprocessing steps (Replicate notebook logic) ---
# Convert categorical columns to Category Datatype
cat_cols = ["CityTier", "ProdTaken", "NumberOfPersonVisited", "NumberOfChildrenVisited", "PreferredPropertyStar", "Passport", "PitchSatisfactionScore", "OwnCar"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

cols_obj = df.select_dtypes(["object"]).columns
for i in cols_obj:
    df[i] = df[i].astype("category")

# Treat Age and MonthlyIncome for missing values
df["MonthlyIncome"] = df.groupby(["Designation"])["MonthlyIncome"].transform(lambda x: x.fillna(x.median()))
df["Age"] = df.groupby(["Designation"])["Age"].transform(lambda x: x.fillna(x.median()))

# Treat other numerical columns for missing values
# Re-identify numerical columns after initial imputation for Age/MonthlyIncome
numerical_cols_for_imputation = df.select_dtypes(include=np.number).columns.tolist()
# Exclude CustomerID, Age, MonthlyIncome as they are already handled or not needed for median imputation here
numerical_cols_for_imputation.remove("CustomerID") # CustomerID is not numeric in the sense of imputation
numerical_cols_for_imputation.remove("MonthlyIncome")
numerical_cols_for_imputation.remove("Age")

for col in numerical_cols_for_imputation:
    df[col] = df[col].fillna(df[col].median())

# Treat other categorical columns for missing values and errors
df["TypeofContact"] = df["TypeofContact"].fillna("Self Enquiry")
df["NumberOfChildrenVisited"] = df["NumberOfChildrenVisited"].fillna(1.0)
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].fillna(3.0)
df.Gender = df.Gender.replace("Fe Male", "Female")

# Separate target variable and drop CustomerID
y = df['ProdTaken']
X_df = df.drop(['CustomerID', 'ProdTaken'], axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X_df, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create directory for processed data
processed_data_dir = "tourism_project/data"
os.makedirs(processed_data_dir, exist_ok=True)

# Save processed data
X_train.to_parquet(os.path.join(processed_data_dir, "X_train.parquet"), index=False)
X_test.to_parquet(os.path.join(processed_data_dir, "X_test.parquet"), index=False)
y_train.to_frame(name='ProdTaken').to_parquet(os.path.join(processed_data_dir, "y_train.parquet"), index=False)
y_test.to_frame(name='ProdTaken').to_parquet(os.path.join(processed_data_dir, "y_test.parquet"), index=False)

# 2. Upload processed dataset to Hugging Face Hub
api = HfApi()
print(f"Uploading processed data to {processed_dataset_repo_id}...")
api.create_repo(repo_id=processed_dataset_repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=processed_data_dir,
    repo_id=processed_dataset_repo_id,
    repo_type="dataset",
    commit_message="Upload processed tourism dataset"
)
print(f"Successfully uploaded processed data to {processed_dataset_repo_id}")

print("Data preparation complete.")
