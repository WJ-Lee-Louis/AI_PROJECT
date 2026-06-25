from pathlib import Path

import numpy as np
from PIL import Image


def convert_cutout_to_binary_mask(
    input_path: Path,
    output_path: Path,
    background_threshold: int,
    fill_holes: bool,
) -> None:
    image = Image.open(input_path).convert("RGBA")
    array = np.array(image)
    rgb = array[..., :3]
    alpha = array[..., 3]

    foreground = (alpha > 0) & (rgb.max(axis=2) > background_threshold)
    mask = (foreground.astype(np.uint8) * 255)

    if fill_holes:
        mask = fill_internal_holes(mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def clean_path_arg(value: str) -> str:
    if not value:
        return value
    return value.strip().strip("\"'\u201c\u201d")


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return ((binary | holes) * 255).astype(np.uint8)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a cut-out or overlay-style mask image into a binary PNG mask."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output binary mask PNG path.")
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=12,
        help="Pixels with all RGB channels at or below this value are treated as background.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill enclosed black holes inside the foreground mask.",
    )
    args = parser.parse_args()

    convert_cutout_to_binary_mask(
        input_path=Path(clean_path_arg(args.input)),
        output_path=Path(clean_path_arg(args.output)),
        background_threshold=args.background_threshold,
        fill_holes=args.fill_holes,
    )
    print(f"Saved binary mask to {Path(clean_path_arg(args.output)).resolve()}")


if __name__ == "__main__":
    main()
