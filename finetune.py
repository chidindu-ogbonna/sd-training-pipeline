""" Script to finetune a model.

```shell
python finetune.py \
    --project_name "my-project" \
    --model_path "" \
    --instance_prompt "" \
    --instance_data_dir "" \
    --with_prior_preservation \
    --prior_preservation_class_prompt "" \
    --num_class_images 200 \
    --train_batch_size 2 \
    --sample_batch_size 2 \
    --learning_rate 2e-06 \
    --max_train_steps 400 \
    --save_steps 200 \
    --train_text_encoder \
    --center_crop \
    --requires_safety_checker \
    --validation_prompt "" \
    --validation_steps 10 \
    --num_validation_images 10
```
"""
import argparse
import itertools
from argparse import Namespace
from slugify import slugify

import accelerate
import torch
from dotenv import load_dotenv

from utils.training import (
    generate_class_images,
    load_model_components,
    training_function,
)

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        default=None,
        type=str,
        required=True,
        help="The name of the project to finetune.",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Model path to load. Can be a local path or a huggingface model path",
    )
    parser.add_argument(
        "--instance_prompt",
        default=None,
        type=str,
        required=True,
        help="The prompt to use to make inference.",
    )
    parser.add_argument(
        "--instance_data_dir",
        default=None,
        type=str,
        required=True,
        help="The directory containing the instance data.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use prior preservation.",
    )
    parser.add_argument(
        "--prior_preservation_class_prompt",
        default=None,
        type=str,
        required=True,
        help="The prompt to use for the prior preservation class data.",
    )
    parser.add_argument(
        "--num_class_images",
        default=200,
        type=int,
        required=False,
        help="The number of class images to generate.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        required=False,
        help="The batch size to use when training. Usually set to 1 if using prior preservation \
            i.e with_prior_preservation is set to true. but not sure if that still \
                applies",
    )
    parser.add_argument(
        "--sample_batch_size",
        default=2,
        type=int,
        required=False,
        help="The batch size to use when generating class images.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        default=1,
        type=int,
        required=False,
        help="The weight to use for the prior loss.",
    )
    parser.add_argument(
        "--class_data_root",
        default="./class_images",
        type=str,
        required=False,
        help="The root directory to save the class images to.",
    )
    # parser.add_argument(
    #     "--train_batch_size",
    #     default=2,
    #     type=int,
    #     required=False,
    #     help="The batch size to use when training. Set to 1 if using prior preservation \
    #         i.e with_prior_preservation is set to true.",
    # )
    parser.add_argument(
        "--learning_rate",
        default=2e-06,
        type=float,
        required=False,
        help="The learning rate to use when training.",
    )
    parser.add_argument(
        "--max_train_steps",
        default=400,
        type=int,
        required=False,
        help="The maximum number of training steps.",
    )
    parser.add_argument(
        "--save_steps",
        default=50,
        type=int,
        required=False,
        help="The number of steps to save the model at.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train the text encoder.",
    )
    parser.add_argument(
        "--center_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to center crop the images.",
    )
    parser.add_argument(
        "--requires_safety_checker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the safety checker. This is required for generating the \
            class images for prior preservation.",
    )
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default=None,
    #     help='Whether training should be resumed from a previous checkpoint. \
    #         Use a path saved by `--save_steps`, or `"latest"` to automatically \
    #         select the last available checkpoint.',
    # )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    args = parser.parse_args()
    print("Arguments parsed...")

    if args.with_prior_preservation:
        generate_class_images(
            model_path=args.model_path,
            class_data_root=args.class_data_root,
            class_prompt=args.prior_preservation_class_prompt,
            num_class_images=args.num_class_images,
            sample_batch_size=args.sample_batch_size,
            requires_safety_checker=args.requires_safety_checker,
        )

    project_name_slug = slugify(args.project_name)
    text_encoder, vae, unet, tokenizer = load_model_components(args.model_path)
    training_args = Namespace(
        pretrained_model_name_or_path=args.model_path,
        resolution=vae.sample_size,  # should be 512
        center_crop=args.center_crop,
        train_text_encoder=args.train_text_encoder,
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        save_steps=args.save_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        mixed_precision="fp16",  # set to "fp16" for mixed-precision training.
        gradient_checkpointing=True,  # set this to True to lower the memory usage.
        use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
        seed=3434554,
        with_prior_preservation=args.with_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        sample_batch_size=2,
        class_data_dir=args.class_data_root,
        class_prompt=args.prior_preservation_class_prompt,
        num_class_images=args.num_class_images,
        lr_scheduler="constant",
        lr_warmup_steps=100,
        output_dir=f"{project_name_slug}-concept",
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        validation_prompt=args.validation_prompt,
        num_validation_images=args.num_validation_images,
        validation_steps=args.validation_steps,
    )
    accelerate.notebook_launcher(
        training_function,
        args=(text_encoder, vae, unet, tokenizer, training_args),
        num_processes=1,
    )
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()
