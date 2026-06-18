from io import BytesIO

import torch
from diffusers.utils import load_image
from PIL import Image

from flux_redux.conditioning import build_redux_condition, prepare_reference_image
from flux_redux.model_loader import load_base_pipeline, load_redux_prior


def load_reference_image(reference_image_url: str, reference_image_bytes: bytes | None):
    if reference_image_bytes is not None:
        return Image.open(BytesIO(reference_image_bytes))
    return load_image(reference_image_url)


def generate_png_bytes(
    prompt: str,
    reference_image_url: str,
    reference_image_bytes: bytes | None,
    prompt_embeds_scale: float,
    pooled_prompt_embeds_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    width: int,
    height: int,
) -> bytes:
    prior = load_redux_prior()
    pipe = load_base_pipeline()

    reference_image = prepare_reference_image(
        load_reference_image(reference_image_url, reference_image_bytes),
        width=width,
        height=height,
    )

    prior_output = build_redux_condition(
        prior=prior,
        reference_image=reference_image,
        prompt=prompt,
        prompt_embeds_scale=prompt_embeds_scale,
        pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        **prior_output,
    ).images[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
