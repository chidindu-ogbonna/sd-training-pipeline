import gc
import itertools
import logging
import math
import os
from argparse import Namespace
from pathlib import Path

import bitsandbytes as bnb
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel  # CLIPTokenizer

from .dataset import DreamBoothDataset, PromptDataset

logger = get_logger(__name__)


def generate_class_images(
    model_path: str,
    class_data_root: str,
    num_class_images: int,
    class_prompt: str,
    sample_batch_size: int,
    requires_safety_checker: bool,
):
    """
    Generate class images for finetuning.
    Usually used if prior preservation is enabled.
    """
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    print("Current class images: ", cur_class_images)

    class_images_exists = cur_class_images == num_class_images
    if class_images_exists:
        print("Class images already exist... skipping generation!")
        return

    if cur_class_images < num_class_images:
        if requires_safety_checker:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                revision="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            # disable safety checker
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                revision="fp16",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to("cuda")

        pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset, batch_size=sample_batch_size
        )

        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                image.save(
                    class_images_dir
                    / f"{example['index'][i] + cur_class_images}.jpg"
                )
        pipeline = None
        gc.collect()
        del pipeline
        with torch.no_grad():
            torch.cuda.empty_cache()

    return


def load_model_components(model_path):
    """Load model components (text_encoder | vae | unet | tokenizer) from model path.

    Args:
        model_path (str): Model path to load. Can be a local path or a
        huggingface model path.
    """
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer"
    )
    return text_encoder, vae, unet, tokenizer


def collate_fn(examples, tokenizer, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # concat class and instance examples for prior preservation
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    # input_ids = torch.cat(input_ids, dim=0)
    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
    ).input_ids
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def log_validation(
    text_encoder,
    tokenizer: AutoTokenizer,
    unet,
    vae,
    args,
    accelerator: Accelerator,
    weight_dtype,
    epoch,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with \
        prompt: {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionPipeline().from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(
                args.validation_prompt,
                num_inference_steps=25,
                generator=generator,
            ).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )
        else:
            print("No tracker found!")
        # if tracker.name == "wandb":
        #     tracker.log(
        #         {
        #             "validation": [
        #                 wandb.Image(
        #                     image, caption=f"{i}: {args.validation_prompt}"
        #                 )
        #                 for i, image in enumerate(images)
        #             ]
        #         }
        #     )

    del pipeline
    torch.cuda.empty_cache()


def restart_from_checkpoint(
    args,
    num_update_steps_per_epoch,
    accelerator: Accelerator,
):
    # TODO: (Promise) Implement and test the restart from checkpoint function
    #  20:07 - 18 Mar, 2023
    """Potentially load in the weights and states from a previous save"""

    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the mos recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new \
                training run."
        )
        args.resume_from_checkpoint = None
        return None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps
        )
        return first_epoch, resume_step


def training_function(text_encoder, vae, unet, tokenizer, args: Namespace):

    logging_dir = Path(args.output_dir, args.logging_dir)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models
    # with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient
    #  accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models
    # is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in \
                distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be \
                supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir
        if args.with_prior_preservation
        else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples, tokenizer, args.with_prior_preservation
        ),
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps
        * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to
    # half-precision as these models are only used for inference, keeping weights in
    # full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    # TODO: (Promise) Disable this for now 17:31 - 18 Mar, 2023
    # vae.decoder.to("cpu")
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training
    # dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num of examples = {len(train_dataset)}")
    print(f"  Length of dataloader: {len(train_dataloader)}")
    print(f"  Num of training epochs: {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = \
            {total_batch_size}"
    )
    print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        result = restart_from_checkpoint(
            args,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            accelerator=accelerator,
        )
        if result:
            first_epoch, resume_step = result
            print(
                f"Resuming from checkpoint {args.restart_from_checkpoint} epoch \
                    {first_epoch} step {resume_step}..."
            )

    for epoch in range(num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each
                #  timestep (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    prediction_type = noise_scheduler.config.prediction_type
                    message = f"Unknown prediction type {prediction_type}"
                    raise ValueError(message)

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute
                    # the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(
                        noise_pred, 2, dim=0
                    )
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = (
                        F.mse_loss(
                            noise_pred.float(),
                            target.float(),
                            reduction="none",
                        )
                        .mean([1, 2, 3])
                        .mean()
                    )

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        noise_pred_prior.float(),
                        target_prior.float(),
                        reduction="mean",
                    )

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(
                        noise_pred.float(), target.float(), reduction="mean"
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the
            # scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.save_steps == 0:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder
                            ),
                        )
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        pipeline.save_pretrained(save_path)

                    if (
                        args.validation_prompt is not None
                        and global_step % args.validation_steps == 0
                    ):
                        log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            epoch,
                        )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            # TODO: (Promise) Configure the tracker first 18:37 - 18 Mar, 2023
            # accelerator.init_trackers("example_project", config=config)
            # https://huggingface.co/docs/accelerate/usage_guides/tracking#integrated-trackers
            # accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                print(
                    f"Reached global step at {global_step}, exiting epoch {epoch}..."
                )
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)

    # TODO: (Promise) Configure the tracker first 19:06 - 18 Mar, 2023
    # accelerator.end_training()
