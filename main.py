# api_key = "55f40bce6be45f1515f35cf30d1c35be"
import requests
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,  # or load_pipeline_from_original_stable_diffusion_ckpt
)


def get_download_url_from_civitai(model_id: int):

    base_url = "https://civitai.com/api/v1/models/"
    url = f"{base_url}{model_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    result = response.json()

    model_versions = result.get("modelVersions")
    first_version = model_versions[0]
    files = first_version.get("files")
    filename = files[0].get("name")
    download_url = files[0].get("downloadUrl")
    return download_url, filename


def download_file(url: str, filename: str):
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            # shutil.copyfileobj(r.raw, f) # Couldn't figure out how to wrap tqdm around this.
            chunk_size = 4096
            for chunk in tqdm(r.iter_content(chunk_size)):
                if chunk:
                    f.write(chunk)
    print(">> File downloaded: ", filename)
    return filename


def convert_model_to_sd(
    checkpoint_path: str, config_file_path: str, from_safetensors: bool
):
    """Convert diffusion model to sd model."""

    if "http" in config_file_path:
        print("Downloading config file path")
        filename = config_file_path.split("/")[-1]
        filename = download_file(config_file_path, filename)
        config_file_path = filename

    print("Loading model!")
        pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=checkpoint_path,
        original_config_file=config_file_path,
        image_size=512,
        prediction_type="epsilon",
        scheduler_type="dpm",
        from_safetensors=from_safetensors,
        device="cuda",
        # EMA weights usually yield higher quality images for inference.
        # Non-EMA weights are usually better to continue fine-tuning.
        extract_ema=False,
    )
    return pipe


if __name__ == "__main__":
    # v1.4 config -> https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    # v1.5 config -> https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    model_id = 4823  # realistic-vision-v12
    url, filename = get_download_url_from_civitai(model_id)
    print("URL retrieved")
    filename = download_file(url, filename)

    pipe = convert_model_to_sd(
        checkpoint_path=filename,
        config_file_path="https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml",
        from_safetensors=True,
    )

    # download_model("realisticVisionV13_v13.safetensors", "civitai", True)
    # pipe = convert_model_to_sd()

    # Make use of the model (this is already a pipe)

    # Fine-tune the model
