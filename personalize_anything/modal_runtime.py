import modal

from shared.modal_runtime import HF_CACHE_VOLUME_NAME


def build_hf_cache_volume() -> modal.Volume:
    return modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def build_personalize_anything_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install(
            "accelerate>=1.0.0",
            "diffusers==0.32.2",
            "huggingface_hub>=0.26.0",
            "numpy>=1.26.0",
            "opencv-python-headless>=4.10.0.84",
            "pillow>=10.4.0",
            "protobuf>=5.0.0",
            "safetensors>=0.4.0",
            "sentencepiece>=0.2.0",
            "torch>=2.4.0",
            "transformers==4.49.0",
        )
        .env(
            {
                "HF_HOME": "/cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .add_local_python_source("shared", "personalize_anything")
    )

