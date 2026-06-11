import os

import torch
from diffusers import FluxPipeline

from xlabs_flux_ip_adapter.config import (
    BASE_MODEL_ID,
    IMAGE_ENCODER_ID,
    IP_ADAPTER_ID,
    IP_ADAPTER_WEIGHT_NAME,
)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Register a Modal Secret named "
            "`huggingface-secret` with HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    return token


def load_pipeline():
    hf_token = get_hf_token()

    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    ).to("cuda")

    pipe.load_ip_adapter(
        IP_ADAPTER_ID,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
        image_encoder_pretrained_model_name_or_path=IMAGE_ENCODER_ID,
        token=hf_token,
    )
    return pipe
