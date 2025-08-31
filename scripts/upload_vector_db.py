"""
- vector_db source files are uploaded into huggingface dataset
- they are downloaded to huggingface space when app is run first time after container build
"""

from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
from src.config import DBConfig


# load env variables from .env file
load_dotenv()

# get hf token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

# init api with token
api = HfApi(token=hf_token)
repo_id = "kenobijr/eu-ai-act-chromadb"

# create repos if not existing yet; get path from DBConfig
config = DBConfig()
folder_path = config.save_dir
folder_path.mkdir(parents=True, exist_ok=True)

api.upload_folder(
    folder_path=folder_path,
    path_in_repo="chroma_db",
    repo_id=repo_id,
    repo_type="dataset",
)
print("Upload complete.")
