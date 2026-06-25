from io import BytesIO

import numpy as np
import torch
from PIL import Image


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_mask_image(mask_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(mask_bytes)).convert("L")


def prepare_reference_image(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)


def mask_to_tensor(mask: Image.Image, latent_w: int, latent_h: int) -> torch.Tensor:
    resized = mask.convert("L").resize((latent_w, latent_h), Image.Resampling.NEAREST)
    mask_array = np.array(resized)
    return torch.tensor(mask_array > 127, dtype=torch.int)


def shift_tensor(tensor: torch.Tensor, shift: int) -> torch.Tensor:
    shifted_tensor = torch.zeros_like(tensor)

    if shift > 0:
        shifted_tensor[:, shift:] = tensor[:, :-shift]
    elif shift < 0:
        shifted_tensor[:, :shift] = tensor[:, -shift:]
    else:
        shifted_tensor = tensor

    return shifted_tensor

