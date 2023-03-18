import os
from pathlib import Path
from typing import Optional
from argparse import Namespace

import huggingface_hub
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    HfFolder,
    Repository,
    create_repo,
    whoami,
)
from slugify import slugify


def save_model_to_hf_hub(
    model_path,
    instance_prompt,
    name_of_concept,
    instance_data_dir,
):
    """Save model to hf hub."""
    HF_TOKEN_WRITE = os.environ.get("HF_TOKEN_WRITE")

    huggingface_hub.login(HF_TOKEN_WRITE)
    hf_api = HfApi()
    username = hf_api.whoami()["name"]

    os.makedirs("fp16_model", exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.save_pretrained("fp16_model")

    repo_id = f"{username}/{slugify(name_of_concept)}"

    readme_text = """Just playing with deliberate"""
    readme_text = f"""---
        license:
        tags:
            - text-to-image
        ---
        ## {name_of_concept}

        `instance_prompt`: **{instance_prompt}**
    """

    # Save the readme to a file
    with open("README.md", "w") as f:
        f.write(readme_text)

    # Save the token identifier to a file
    with open("token_identifier.txt", "w") as f:
        f.write(instance_prompt)

    create_repo(repo_id, private=True, token=HF_TOKEN_WRITE)
    operations = [
        CommitOperationAdd(
            path_in_repo="token_identifier.txt",
            path_or_fileobj="token_identifier.txt",
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj="README.md",
        ),
    ]

    hf_api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload the concept {name_of_concept} embeds and token",
        token=HF_TOKEN_WRITE,
    )
    hf_api.upload_folder(
        folder_path="fp16_model",
        path_in_repo="",
        repo_id=repo_id,
        token=HF_TOKEN_WRITE,
    )
    hf_api.upload_folder(
        folder_path=instance_data_dir,
        path_in_repo="concept_images",
        repo_id=repo_id,
        token=HF_TOKEN_WRITE,
    )
    print(
        f"Your concept was saved successfully. [Click here to access it](https://huggingface.co/{repo_id})"
    )


def get_full_repo_name(
    model_id: str,
    organization: Optional[str] = None,
    token: Optional[str] = None,
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def push_to_hub(args: Namespace):

    if args.hub_model_id is None:
        repo_name = get_full_repo_name(
            Path(args.output_dir).name, token=args.hub_token
        )
    else:
        repo_name = args.hub_model_id
    create_repo(repo_name, exist_ok=True, token=args.hub_token)
    repo = Repository(
        args.output_dir,
        clone_from=repo_name,
        token=args.hub_token,
    )

    with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
            gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
            gitignore.write("epoch_*\n")

    repo.push_to_hub(
        commit_message="End of training",
        blocking=False,
        auto_lfs_prune=True,
    )


if __name__ == "__main__":
    name_of_concept = "training-deliberate"
    model_path = "deliberate"
    instance_data_dir = ""
    instance_prompt = "promise"

    save_model_to_hf_hub(
        model_path,
        instance_prompt,
        name_of_concept,
        instance_data_dir,
    )
