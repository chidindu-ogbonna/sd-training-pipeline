from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)


def _download_from_civitai(name: str):
    pass


def _download_from_hf(name: str):
    pass


def download_model(
    name: str, repository: str, from_safe_tensors: bool = False
):
    """Get model from repository.

    Args:
        name (str): name of the model
        repository (str): which repository to use 'civitai' or 'huggingface'
    """
    if repository == "civitai":
        _download_from_civitai(name)
    elif repository == "huggingface":
        _download_from_hf(name)
    else:
        raise ValueError(f"Repository {repository} not supported")


def fine_tune_model():
    pass


def convert_model_to_sd():
    """Convert diffusion model to sd model."""
    print("Loading model!")
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path="realisticVisionV13_v13.safetensors",
        original_config_file="v1-inference.yaml",
        image_size=512,
        prediction_type="epsilon",
        scheduler_type="dpm",
        from_safetensors=True,
        device="cpu",
        # EMA weights usually yield higher quality images for inference.
        # Non-EMA weights are usually better to continue fine-tuning.
        extract_ema=False,
    )
    return pipe


if __name__ == "__main__":
    download_model("realisticVisionV13_v13.safetensors", "civitai", True)
    pipe = convert_model_to_sd()

    # Make use of the model (this is already a pipe)

    # Fine-tune the model
