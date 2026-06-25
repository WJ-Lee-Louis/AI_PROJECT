from io import BytesIO

import torch
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from personalize_anything.conditioning import (
    load_mask_image,
    load_rgb_image,
    mask_to_tensor,
    prepare_reference_image,
    shift_tensor,
)
from personalize_anything.model_loader import add_third_party_path, load_pipeline


def generate_png_bytes(
    reference_image_bytes: bytes,
    mask_image_bytes: bytes,
    reference_prompt: str,
    prompt: str,
    inversion_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    width: int,
    height: int,
    tau: float,
    eta: float,
    gamma: float,
    start_timestep: float,
    stop_timestep: float,
    mask_shift: int,
) -> bytes:
    add_third_party_path()

    from attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor

    pipe = load_pipeline()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    reference_image = prepare_reference_image(
        load_rgb_image(reference_image_bytes),
        width=width,
        height=height,
    )

    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * latent_w

    mask = mask_to_tensor(load_mask_image(mask_image_bytes), latent_w=latent_w, latent_h=latent_h)
    shift_mask = shift_tensor(mask, mask_shift)

    # Inversion should run with the original FLUX attention processor.
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: FluxAttnProcessor2_0(),
    )
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        source_prompt=inversion_prompt,
        image=reference_image,
        height=height,
        width=width,
        num_inversion_steps=num_inference_steps,
        gamma=gamma,
        generator=generator,
    )

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name,
            tau=tau,
            mask=mask,
            shift_mask=shift_mask,
            device=device,
            img_dims=img_dims,
            concept_process=False,
        ),
    )

    result = pipe(
        [reference_prompt, prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        start_timestep=start_timestep,
        stop_timestep=stop_timestep,
        num_inference_steps=num_inference_steps,
        eta=eta,
        generator=generator,
    ).images[1]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()

