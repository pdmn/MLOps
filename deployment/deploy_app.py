import os
from huggingface_hub import HfApi, login

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

# Define your Hugging Face username and Space repo name
hf_username = os.environ.get('HF_USERNAME', 'your_hf_username_here')  # Placeholder for username
space_repo_id = f"{hf_username}/tourism-prediction-app" # This should be your Hugging Face Space name

# Define the path to the directory containing the Streamlit app and requirements
deployment_folder = "tourism_project/deployment"

# Initialize Hugging Face API
api = HfApi()

# Create the Hugging Face Space repository if it doesn't exist
# Make sure the Space is set to 'streamlit' SDK
try:
    api.create_repo(repo_id=space_repo_id,
                    repo_type="space",
                    space_sdk="streamlit",
                    exist_ok=True)
    print(f"Ensured Hugging Face Space '{space_repo_id}' exists.")
except Exception as e:
    print(f"Error creating/checking Hugging Face Space: {e}")
    exit()

# Upload the contents of the deployment folder to the Hugging Face Space
try:
    print(f"Uploading deployment files to Hugging Face Space '{space_repo_id}'...")
    api.upload_folder(
        folder_path=deployment_folder,
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Deploy Streamlit app and requirements"
    )
    print(f"Successfully uploaded deployment files to '{space_repo_id}'.")
except Exception as e:
    print(f"Failed to upload deployment files: {e}")
