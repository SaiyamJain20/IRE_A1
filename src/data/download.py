import os
import subprocess
import zipfile
import urllib.request
from urllib.error import URLError

URLS = {
    "MINDsmall_train": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "MINDsmall_dev": "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
    "ebnerd_demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "ebnerd_small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip"
}

def download_and_extract(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    
    for name, url in URLS.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")
        extract_dir = os.path.join(data_dir, name)
        
        # Download
        if not os.path.exists(zip_path) and not os.path.exists(extract_dir):
            print(f"Downloading {name}...")
            try:
                subprocess.run(["wget", "-q", "--show-progress", "-O", zip_path, url], check=True)
            except Exception as e:
                print(f"wget failed, trying urllib: {e}")
                urllib.request.urlretrieve(url, zip_path)
        else:
            print(f"File {zip_path} already exists or is extracted.")
            
        # Extract
        if os.path.exists(zip_path) and not os.path.exists(extract_dir):
            print(f"Extracting {name}...")
            # For MIND, files are loose in the zip, so extract to a named folder
            # For EB-NeRD, it creates a folder automatically, but zipfile extraction handles it
            if "MIND" in name:
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                    
            print(f"Extracted {name}.")

if __name__ == "__main__":
    download_and_extract()
