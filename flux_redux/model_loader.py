import os

import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline

from flux_redux.config import BASE_MODEL_ID, REDUX_MODEL_ID


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Register a Modal Secret named "
            "`huggingface-secret` with HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    return token


def load_redux_prior():
    return FluxPriorReduxPipeline.from_pretrained(
        REDUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    ).to("cuda")


def load_base_pipeline():
    return FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    ).to("cuda")
