""" Script to finetune a model.

python finetune.py \
 --project_name "my-project" \
 --model_path " \
 --instance_prompt "" \
 --with_prior_preservation True \
 --prior_preservation_class_prompt "" \
 --num_class_images 200 \
 --sample_batch_size 2 \
 --learning_rate 2e-06 \
 --max_train_steps 400 \
 --save_steps 50 \
 --train_text_encoder False \
 --center_crop True
"""
import argparse
import itertools
from argparse import Namespace

import accelerate
import torch

from utils.training import (
    generate_class_images,
    load_model_components,
    training_function,
)

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
        "--with_prior_preservation",
        default=False,
        type=bool,
        required=False,
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
        default=False,
        type=bool,
        required=False,
        help="Whether to train the text encoder.",
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        type=bool,
        required=False,
        help="Whether to center crop the images.",
    )
    args = parser.parse_args()

    if args.with_prior_preservation:
        train_batch_size = 1
    else:
        train_batch_size = 2

    if args.with_prior_preservation:
        generate_class_images(
            model_path=args.model_path,
            class_data_root=args.class_data_root,
            class_prompt=args.prior_preservation_class_prompt,
            num_class_images=args.num_class_images,
            sample_batch_size=args.sample_batch_size,
        )

    text_encoder, vae, unet, tokenizer = load_model_components(args.model_path)
    training_args = Namespace(
        pretrained_model_name_or_path=args.model_path,
        resolution=vae.sample_size,  # should be 512
        center_crop=args.center_crop,
        train_text_encoder=args.train_text_encoder,
        instance_data_dir=args.project_name,
        instance_prompt=args.instance_prompt,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        save_steps=args.save_steps,
        train_batch_size=train_batch_size,
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
        output_dir=f"{args.project_name}-concept",
    )
    accelerate.notebook_launcher(
        training_function,
        args=(text_encoder, vae, unet, tokenizer, args),
    )
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()
