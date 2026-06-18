from PIL import Image, ImageOps


def prepare_reference_image(reference_image: Image.Image, width: int, height: int) -> Image.Image:
    image = reference_image.convert("RGB")
    return ImageOps.contain(image, (width, height))


def build_redux_condition(
    prior,
    reference_image: Image.Image,
    prompt: str,
    prompt_embeds_scale: float,
    pooled_prompt_embeds_scale: float,
):
    kwargs = {
        "image": reference_image,
        "prompt_embeds_scale": prompt_embeds_scale,
        "pooled_prompt_embeds_scale": pooled_prompt_embeds_scale,
    }
    if prompt:
        kwargs["prompt"] = prompt

    return prior(**kwargs)
