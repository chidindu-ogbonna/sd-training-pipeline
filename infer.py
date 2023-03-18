"""Script for inference on a model.

```shell
python infer.py \
    --model_path "" \
    --scheduler "" \
    --prompt "A photo of a" \
    --negative_prompt "A photo of a" \
    --guidance_scale 9 \
    --num_inference_steps 25 \
    --num_images_per_prompt 4 \
    --num_rows 1 \
    --requires_safety_checker
```
"""

import argparse
import diffusers
from dotenv import load_dotenv

from utils.inference import load_model, infer

load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path to the model to load. Can be a local path or a huggingface model \
            path.",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        required=False,
        help="The scheduler to use for inference. If not provided, the model's default \
            scheduler is used",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        required=True,
        help="The prompt to use for inference.",
    )
    parser.add_argument(
        "--negative_prompt",
        default=None,
        type=str,
        required=False,
        help="The negative prompt to use for inference.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=9,
        help="The guidance scale to use for inference.",
    )

    parser.add_argument(
        "--num_inference_steps",
        default=25,
        type=int,
        required=False,
        help="The number of inference steps to use.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=4,
        type=int,
        required=False,
        help="The number of images to generate per prompt.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1,
        help="The number of rows to use for image grid.",
    )
    parser.add_argument(
        "--requires_safety_checker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the safety checker for inference.",
    )

    args = parser.parse_args()

    scheduler = None
    if args.scheduler:
        try:
            scheduler = getattr(diffusers, args.scheduler)
        except AttributeError as e:
            print(e)
            print("Using the default scheduler")

    pipeline = load_model(
        args.model_path,
        scheduler=scheduler,
        requires_safety_checker=args.requires_safety_checker,
    )
    grid = infer(
        pipeline,
        prompt=args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_rows=args.num_rows,
        negative_prompt=args.negative_prompt,
    )
    grid.show()
