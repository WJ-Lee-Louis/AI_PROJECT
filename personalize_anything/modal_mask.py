from pathlib import Path

import modal

from personalize_anything.config import (
    DEFAULT_MASK_OUTPUT_DIR,
    DEFAULT_MASK_OUTPUT_PATH,
    DEFAULT_REFERENCE_IMAGE_PATH,
    DEFAULT_SEGMENTATION_LABELS,
    MASK_GPU,
    MASK_APP_NAME,
)
from personalize_anything.io_utils import clean_path_arg, resolve_mask_output_path
from personalize_anything.modal_runtime import build_hf_cache_volume, build_personalize_anything_image
from shared.modal_runtime import HF_SECRET_NAME


app = modal.App(MASK_APP_NAME)
hf_cache = build_hf_cache_volume()
image = build_personalize_anything_image()


@app.function(
    image=image,
    gpu=MASK_GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=1200,
)
def generate_mask_with_grounded_sam(
    image_bytes: bytes,
    labels: str,
    threshold: float = 0.3,
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    segmenter_id: str = "facebook/sam-vit-base",
    max_detections: int = 1,
    polygon_refinement: bool = True,
) -> dict:
    from personalize_anything.mask_generation import generate_grounded_sam_mask

    return generate_grounded_sam_mask(
        image_bytes=image_bytes,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id,
        segmenter_id=segmenter_id,
        max_detections=max_detections,
        polygon_refinement=polygon_refinement,
    )


@app.local_entrypoint()
def main(
    reference_image_path: str = DEFAULT_REFERENCE_IMAGE_PATH,
    labels: str = DEFAULT_SEGMENTATION_LABELS,
    threshold: float = 0.3,
    detector_id: str = "IDEA-Research/grounding-dino-tiny",
    segmenter_id: str = "facebook/sam-vit-base",
    max_detections: int = 1,
    polygon_refinement: bool = True,
    output_path: str = DEFAULT_MASK_OUTPUT_PATH,
):
    reference_path = Path(clean_path_arg(reference_image_path))
    image_bytes = reference_path.read_bytes()

    print(f"Using local reference image: {reference_path.resolve()}")
    print(f"Using segmentation labels: {labels}")

    result = generate_mask_with_grounded_sam.remote(
        image_bytes=image_bytes,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id,
        segmenter_id=segmenter_id,
        max_detections=max_detections,
        polygon_refinement=polygon_refinement,
    )

    path = resolve_mask_output_path(
        output_path=output_path,
        default_dir=DEFAULT_MASK_OUTPUT_DIR,
        image_path=str(reference_path),
        labels=labels,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result["mask_png_bytes"])

    print(f"Saved mask to {path.resolve()}")
    for index, detection in enumerate(result["detections"]):
        print(f"Detection {index}: {detection}")
