import requests
from tqdm import tqdm

try:
    # XXX: Depending on the version of the diffusers used (not sure which one :))
    # you should use "load_pipeline_from_original_stable_diffusion_ckpt"
    # instead of "download_from_original_stable_diffusion_ckpt"
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )
except ImportError:
    print(
        'Error importing "download_from_original_stable_diffusion_ckpt".\n \
            Using "load_pipeline_from_original_stable_diffusion_ckpt" instead.'
    )
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        load_pipeline_from_original_stable_diffusion_ckpt as download_from_original_stable_diffusion_ckpt,
    )


def download_file(url: str, filename: str):
    """Deprecated in favour of [`transformers.utils.hub.download_url`]."""
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            # XXX: Couldn't figure out how to wrap tqdm around this.
            # shutil.copyfileobj(r.raw, f)
            chunk_size = 4096
            for chunk in tqdm(r.iter_content(chunk_size)):
                if chunk:
                    f.write(chunk)
    print("File downloaded: ", filename)
    return filename


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_file_path: str,
    from_safetensors: bool,
    use_controlnet: bool = False,
):
    """Convert checkpoint to stable diffusion model."""

    if "http" in config_file_path:
        print("Downloading config file path...")
        filename = config_file_path.split("/")[-1]
        filename = download_file(config_file_path, filename)
        config_file_path = filename

    print("Loading model...")
    pipeline = download_from_original_stable_diffusion_ckpt(
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
        controlnet=use_controlnet,
    )
    return pipeline
