"""Script to save a model to hf hub.

python save_model_to_hf.py \
    --model_path "" \
    --instance_prompt "" \
    --name_of_concept "" \
    --instance_data_dir ""
"""

import argparse

from dotenv import load_dotenv

from utils.hf import save_model_to_hf_hub

load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path to the model to load. Must be a local path.",
    )
    parser.add_argument(
        "--instance_prompt",
        default=None,
        type=str,
        required=True,
        help="The prompt to use to make inference.",
    )
    parser.add_argument(
        "--name_of_concept",
        default=None,
        type=str,
        required=True,
        help="The name of the concept you are finetuning. This will be the name of the \
            model on the HF Hub.",
    )
    parser.add_argument(
        "--instance_data_dir",
        default=None,
        type=str,
        required=True,
        help="The directory containing the instance data.",
    )
    args = parser.parse_args()

    save_model_to_hf_hub(
        model_path=args.model_path,
        instance_prompt=args.instance_prompt,
        name_of_concept=args.name_of_concept,
        instance_data_dir=args.instance_data_dir,
    )
