import os

import huggingface_hub
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from slugify import slugify


name_of_your_concept = "training-deliberate"
hf_token_write = "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"
output_dir = "deliberate"
save_path = ""

if __name__ == "__main__":
    huggingface_hub.login(hf_token_write)
    api = HfApi()
    your_username = api.whoami()["name"]
    pipe = StableDiffusionPipeline.from_pretrained(
        output_dir, torch_dtype=torch.float16
    ).to("cuda")
    os.makedirs("fp16_model", exist_ok=True)
    pipe.save_pretrained("fp16_model")

    repo_id = f"{your_username}/{slugify(name_of_your_concept)}"
    # output_dir = args.output_dir
    hf_token = hf_token_write

    if save_path:
        images_upload = os.listdir(save_path)
        image_string = ""
        # repo_id = f"sd-dreambooth-library/{slugify(name_of_your_concept)}"
        for i, image in enumerate(images_upload):
            image_string = f"""{image_string}![image {i}](https://huggingface.co/{repo_id}/resolve/main/concept_images/{image})"""

    readme_text = """Just playing with deliberate"""
    # readme_text = f"""---
    # license: creativeml-openrail-m
    # tags:
    # - text-to-image
    # ---
    # ### {name_of_your_concept} on Stable Diffusion via Dreambooth
    # #### model by {api.whoami()["name"]}
    # This your the Stable Diffusion model fine-tuned the {name_of_your_concept} concept taught to Stable Diffusion with Dreambooth.
    # It can be used by modifying the `instance_prompt`: **{instance_prompt}**

    # You can also train your own concepts and upload them to the library by using [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb).
    # And you can run your new concept via `diffusers`: [Colab Notebook for Inference](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_inference.ipynb), [Spaces with the Public Concepts loaded](https://huggingface.co/spaces/sd-dreambooth-library/stable-diffusion-dreambooth-concepts)

    # Here are the images used for training this concept:
    # {image_string}
    # """

    # Save the readme to a file
    readme_file = open("README.md", "w")
    readme_file.write(readme_text)
    readme_file.close()

    # Save the token identifier to a file
    text_file = open("token_identifier.txt", "w")
    text_file.write("anything you want")
    text_file.close()
    operations = [
        CommitOperationAdd(
            path_in_repo="token_identifier.txt",
            path_or_fileobj="token_identifier.txt",
        ),
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj="README.md"
        ),
    ]
    create_repo(repo_id, private=True, token=hf_token)

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload the concept {name_of_your_concept} embeds and token",
        token=hf_token,
    )
    api.upload_folder(
        folder_path="fp16_model",
        path_in_repo="",
        repo_id=repo_id,
        token=hf_token,
    )
    # api.upload_folder(
    #     folder_path=save_path,
    #     path_in_repo="concept_images",
    #     repo_id=repo_id,
    #     token=hf_token,
    # )
    print(
        f"Your concept was saved successfully. [Click here to access it](https://huggingface.co/{repo_id})"
    )
