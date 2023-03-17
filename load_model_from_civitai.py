"""Script to load a model from Civitai.

model w/ id 4823 => realistic-vision-v12

python load_model_from_civitai.py \
    --civitai_model_id 4823 \
    --config_file_path "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml" \
    --save_to_path True
"""

import argparse

from dotenv import load_dotenv

from utils.civitai import load_model_from_civitai

load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--civitai_model_id",
        default=None,
        type=int,
        required=True,
        help="The id of the model to load from Civitai.",
    )
    parser.add_argument(
        "--config_file_path",
        default=None,
        type=str,
        required=True,
        help="The path to the config file to use.",
    )
    parser.add_argument(
        "--save_to_path",
        default=False,
        type=bool,
        required=False,
        help="Whether to save the model to a path.",
    )
    args = parser.parse_args()
    pipeline = load_model_from_civitai(
        args.civitai_model_id,
        config_file_path=args.config_file_path,
        save_to_path=True,
    )
    print("Pipeline loaded successfully.")
