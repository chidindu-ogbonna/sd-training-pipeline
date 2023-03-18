import os
from typing import List, Optional, Union

import huggingface_hub
import torch
import torch.utils.checkpoint
from diffusers import SchedulerMixin, StableDiffusionPipeline
from diffusers.configuration_utils import ConfigMixin

from utils.images import image_grid


def load_model(
    model_path: str,
    scheduler: Optional[Union[SchedulerMixin, ConfigMixin]],
    requires_safety_checker: bool = True,
):
    """Load model pipeline from model path.

    Args:
        model_path (str): Model path to load. Can be a local path or
        a huggingface model path.
        scheduler (SchedulerMixin): Scheduler to load.
        requires_safety_checker (bool): Whether to load the safety checker.

    ```py
        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> # or
        >>> scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        >>> # set the scheduler
        >>> pipeline.scheduler = scheduler
    ```
    """
    HF_TOKEN_READ = os.environ.get("HF_TOKEN_READ")
    huggingface_hub.login(HF_TOKEN_READ)
    if requires_safety_checker:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
    else:
        # disable safety checker
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")

    if scheduler:
        pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    return pipeline


def infer(
    pipeline: StableDiffusionPipeline,
    prompt,
    num_images_per_prompt,
    num_inference_steps,
    guidance_scale,
    num_rows,
    negative_prompt: Optional[Union[str, List[str]]] = None,
):
    """Infer images from a prompt."""

    all_images = []
    for _ in range(num_rows):
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        ).images
        all_images.extend(images)

    # torch.cuda.empty_cache()
    grid = image_grid(all_images, num_rows, num_images_per_prompt)
    return grid


if __name__ == "__main__":
    model_path = "stabilityai/stable-diffusion-2-1-base"
    prompt = "a black healthtracka waterbottle on a white table intricate detail, hi-res photography"
    num_samples = 8
    num_rows = 1
    guidance_scale = 9
    num_inference_steps = 125
    negative_prompt = None

    pipeline = load_model(model_path, scheduler=None)
    grid = infer(
        pipeline,
        prompt,
        num_samples,
        num_inference_steps,
        guidance_scale,
        num_rows,
        negative_prompt=negative_prompt,
    )
    grid
