import pandas as pd
from huggingface_hub import HfApi, login
import os

# Get Hugging Face token from environment variables
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
    login(token=HF_TOKEN)
except KeyError:
    print("HF_TOKEN environment variable not set. Please set it to push to Hugging Face Hub.")
    exit()
except Exception as e:
    print(f"Error logging into Hugging Face: {e}")
    exit()

# Define the path to the dataset
dataset_path = "tourism.csv"

# Ensure the dataset file exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}")
    exit()

# Initialize Hugging Face API
api = HfApi()

# Define your Hugging Face username and dataset repo name
# Replace <YOUR_HF_USERNAME> with your actual Hugging Face username
hf_username = os.environ.get('HF_USERNAME', 'your_hf_username_here')
dataset_repo_id = f"{hf_username}/tourism-raw-dataset"

# Upload the dataset to Hugging Face Hub
try:
    print(f"Uploading {dataset_path} to {dataset_repo_id}...")
    api.upload_file(
        path_or_fileobj=dataset_path,
        path_in_repo=dataset_path,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message="Upload raw tourism dataset"
    )
    print(f"Successfully uploaded {dataset_path} to {dataset_repo_id}")
except Exception as e:
    print(f"Failed to upload dataset: {e}")
