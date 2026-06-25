from pathlib import Path

import modal

from personalize_anything.config import (
    APP_NAME,
    DEFAULT_INVERSION_PROMPT,
    DEFAULT_MASK_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT,
    DEFAULT_REFERENCE_IMAGE_PATH,
    DEFAULT_REFERENCE_PROMPT,
)
from personalize_anything.io_utils import clean_path_arg, resolve_image_output_path
from personalize_anything.modal_runtime import build_hf_cache_volume, build_personalize_anything_image
from shared.modal_runtime import DEFAULT_GPU, HF_SECRET_NAME


app = modal.App(APP_NAME)
hf_cache = build_hf_cache_volume()
image = build_personalize_anything_image()


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=2400,
)
def generate_with_personalize_anything(
    reference_image_bytes: bytes,
    mask_image_bytes: bytes,
    reference_prompt: str,
    prompt: str,
    inversion_prompt: str = "",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    seed: int = 18,
    width: int = 1024,
    height: int = 1024,
    tau: float = 0.6,
    eta: float = 1.0,
    gamma: float = 1.0,
    start_timestep: float = 0.0,
    stop_timestep: float = 0.99,
    mask_shift: int = 0,
) -> bytes:
    from personalize_anything.inference import generate_png_bytes

    return generate_png_bytes(
        reference_image_bytes=reference_image_bytes,
        mask_image_bytes=mask_image_bytes,
        reference_prompt=reference_prompt,
        prompt=prompt,
        inversion_prompt=inversion_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
        tau=tau,
        eta=eta,
        gamma=gamma,
        start_timestep=start_timestep,
        stop_timestep=stop_timestep,
        mask_shift=mask_shift,
    )


@app.local_entrypoint()
def main(
    reference_image_path: str = DEFAULT_REFERENCE_IMAGE_PATH,
    mask_path: str = DEFAULT_MASK_PATH,
    reference_prompt: str = DEFAULT_REFERENCE_PROMPT,
    prompt: str = DEFAULT_PROMPT,
    inversion_prompt: str = DEFAULT_INVERSION_PROMPT,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    seed: int = 18,
    width: int = 1024,
    height: int = 1024,
    tau: float = 0.6,
    eta: float = 1.0,
    gamma: float = 1.0,
    start_timestep: float = 0.0,
    stop_timestep: float = 0.99,
    mask_shift: int = 0,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    reference_path = Path(clean_path_arg(reference_image_path))
    mask_file_path = Path(clean_path_arg(mask_path))

    print(f"Using local reference image: {reference_path.resolve()}")
    print(f"Using local mask image: {mask_file_path.resolve()}")
    print(f"Using reference prompt: {reference_prompt}")
    print(f"Using target prompt: {prompt}")
    if inversion_prompt:
        print(f"Using inversion prompt: {inversion_prompt}")
    else:
        print("Using empty inversion prompt.")

    output_bytes = generate_with_personalize_anything.remote(
        reference_image_bytes=reference_path.read_bytes(),
        mask_image_bytes=mask_file_path.read_bytes(),
        reference_prompt=reference_prompt,
        prompt=prompt,
        inversion_prompt=inversion_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
        tau=tau,
        eta=eta,
        gamma=gamma,
        start_timestep=start_timestep,
        stop_timestep=stop_timestep,
        mask_shift=mask_shift,
    )

    path = resolve_image_output_path(
        output_path=output_path,
        default_dir=DEFAULT_OUTPUT_DIR,
        reference_image_path=str(reference_path),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output_bytes)
    print(f"Saved image to {path.resolve()}")

