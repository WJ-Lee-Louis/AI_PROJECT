import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

from personalize_anything.config import BASE_MODEL_ID


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Register a Modal Secret named "
            "`huggingface-secret` with HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    return token


def add_third_party_path() -> Path:
    third_party_dir = Path(__file__).resolve().parent / "third_party"
    third_party_path = str(third_party_dir)
    if third_party_path not in sys.path:
        sys.path.insert(0, third_party_path)
    return third_party_dir


@lru_cache(maxsize=1)
def load_pipeline():
    add_third_party_path()

    from pipeline import RFInversionParallelFluxPipeline

    return RFInversionParallelFluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        token=get_hf_token(),
    ).to("cuda")

