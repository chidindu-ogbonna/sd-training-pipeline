import os

import requests
from slugify import slugify

from .utils import download_file, load_model_from_checkpoint


def download_model_from_civitai(model_id: int):
    url, filename, model_name = get_download_url_from_civitai(model_id)
    print(f"Downloading file {url}...")
    filename = download_file(url, filename)
    return filename, model_name


def get_download_url_from_civitai(model_id: int):

    base_url = "https://civitai.com/api/v1/models/"
    url = f"{base_url}{model_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    result = response.json()
    model_name = result["namem"]

    model_versions = result.get("modelVersions")
    first_version = model_versions[0]
    files = first_version.get("files")
    filename = files[0].get("name")
    download_url = files[0].get("downloadUrl")
    return download_url, filename, model_name


def load_model_from_civitai(
    model_id: int,
    config_file_path: str,
    save_to_path: bool,
):
    file_location, model_name = download_model_from_civitai(model_id)
    pipeline = load_model_from_checkpoint(
        checkpoint_path=file_location,
        config_file_path=config_file_path,
        from_safetensors=True,
    )
    if save_to_path:
        model_path = slugify(model_name)
        os.makedirs(model_path, exist_ok=True)
        pipeline.save_pretrained(model_path)

    return pipeline
