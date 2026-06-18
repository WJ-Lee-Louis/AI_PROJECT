from pathlib import Path
from urllib.parse import urlparse

import modal

from flux_redux.config import (
    APP_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REFERENCE_IMAGE_PATH,
    DEFAULT_REFERENCE_IMAGE_URL,
)
from shared.modal_runtime import (
    DEFAULT_GPU,
    HF_SECRET_NAME,
    build_diffusers_image,
    build_hf_cache_volume,
)


app = modal.App(APP_NAME)
hf_cache = build_hf_cache_volume()
image = build_diffusers_image()


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available output path for {path}")


def reference_stem(reference_image_path: str, reference_image_url: str) -> str:
    if reference_image_path:
        return Path(reference_image_path).stem

    url_path = urlparse(reference_image_url).path
    stem = Path(url_path).stem
    return stem or "reference"


def resolve_output_path(
    output_path: str,
    reference_image_path: str,
    reference_image_url: str,
) -> Path:
    if output_path:
        return Path(output_path)

    filename = f"{reference_stem(reference_image_path, reference_image_url)}_generated.png"
    return next_available_path(Path(DEFAULT_OUTPUT_DIR) / filename)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=1800,
)
def generate_with_redux(
    prompt: str,
    reference_image_url: str,
    reference_image_bytes: bytes | None = None,
    prompt_embeds_scale: float = 1.0,
    pooled_prompt_embeds_scale: float = 1.0,
    guidance_scale: float = 2.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    from flux_redux.inference import generate_png_bytes

    return generate_png_bytes(
        prompt=prompt,
        reference_image_url=reference_image_url,
        reference_image_bytes=reference_image_bytes,
        prompt_embeds_scale=prompt_embeds_scale,
        pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
    )


@app.local_entrypoint()
def main(
    prompt: str = "",
    reference_image_url: str = DEFAULT_REFERENCE_IMAGE_URL,
    reference_image_path: str = DEFAULT_REFERENCE_IMAGE_PATH,
    prompt_embeds_scale: float = 1.0,
    pooled_prompt_embeds_scale: float = 1.0,
    guidance_scale: float = 2.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    reference_image_bytes = None
    if reference_image_path:
        reference_path = Path(reference_image_path)
        reference_image_bytes = reference_path.read_bytes()
        print(f"Using local reference image: {reference_path.resolve()}")
    else:
        print(f"Using reference image URL: {reference_image_url}")

    if prompt:
        print(f"Using text prompt: {prompt}")
    else:
        print("Using empty text prompt.")

    output_bytes = generate_with_redux.remote(
        prompt=prompt,
        reference_image_url=reference_image_url,
        reference_image_bytes=reference_image_bytes,
        prompt_embeds_scale=prompt_embeds_scale,
        pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
    )

    path = resolve_output_path(
        output_path=output_path,
        reference_image_path=reference_image_path,
        reference_image_url=reference_image_url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output_bytes)
    print(f"Saved image to {path.resolve()}")
