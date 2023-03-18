import os

import huggingface_hub
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import CommitOperationAdd, HfApi, create_repo
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
