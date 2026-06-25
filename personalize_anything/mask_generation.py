from io import BytesIO

import numpy as np
import torch
from PIL import Image

from personalize_anything.model_loader import add_third_party_path


def parse_labels(labels: str) -> list[str]:
    return [label.strip() for label in labels.replace(";", ",").split(",") if label.strip()]


def detection_to_dict(detection) -> dict:
    box = detection.box
    return {
        "label": detection.label,
        "score": None if detection.score is None else float(detection.score),
        "box": None
        if box is None
        else {
            "xmin": int(box.xmin),
            "ymin": int(box.ymin),
            "xmax": int(box.xmax),
            "ymax": int(box.ymax),
        },
    }


def generate_grounded_sam_mask(
    image_bytes: bytes,
    labels: str,
    threshold: float,
    detector_id: str,
    segmenter_id: str,
    max_detections: int,
    polygon_refinement: bool,
) -> dict:
    add_third_party_path()

    from grounding_sam import grounded_segmentation, prepare_model

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    object_detector, processor, segmentator = prepare_model(
        device=device,
        detector_id=detector_id,
        segmenter_id=segmenter_id,
    )

    _, detections, _ = grounded_segmentation(
        object_detector=object_detector,
        processor=processor,
        segmentator=segmentator,
        image=image,
        labels=parse_labels(labels),
        threshold=threshold,
        polygon_refinement=polygon_refinement,
    )

    detections = [detection for detection in detections if detection.mask is not None]
    detections.sort(key=lambda detection: detection.score or 0.0, reverse=True)
    if max_detections > 0:
        detections = detections[:max_detections]

    if not detections:
        raise RuntimeError(
            f"No segmentation mask was generated for labels={labels!r}. "
            "Try a simpler label such as 'person', lower --threshold, or use a UI mask."
        )

    mask = np.zeros(image.size[::-1], dtype=np.uint8)
    for detection in detections:
        mask[np.asarray(detection.mask) > 0] = 255

    mask_image = Image.fromarray(mask, mode="L")
    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")

    return {
        "mask_png_bytes": buffer.getvalue(),
        "detections": [detection_to_dict(detection) for detection in detections],
    }

