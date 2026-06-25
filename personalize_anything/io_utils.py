from pathlib import Path


def clean_path_arg(value: str) -> str:
    if not value:
        return value
    return value.strip().strip("\"'\u201c\u201d")


def safe_label_stem(labels: str) -> str:
    cleaned = labels.replace(",", "_").replace(";", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in ("_", "-"))
    return cleaned.strip("_") or "subject"


def avoid_overwrite_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = path.suffix or ".png"
    base = path.with_suffix(suffix)

    for index in range(1, 1000):
        candidate = base.with_name(f"{base.stem}_{index:02d}{suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available output path for {path}")


def resolve_image_output_path(output_path: str, default_dir: str, reference_image_path: str) -> Path:
    reference_stem = Path(reference_image_path).stem or "reference"
    filename = f"{reference_stem}_personalized.png"

    if output_path:
        path = Path(clean_path_arg(output_path))
        if path.suffix:
            return avoid_overwrite_path(path)
        return avoid_overwrite_path(path / filename)

    return avoid_overwrite_path(Path(default_dir) / filename)


def resolve_mask_output_path(output_path: str, default_dir: str, image_path: str, labels: str) -> Path:
    image_stem = Path(image_path).stem or "reference"
    filename = f"{image_stem}_{safe_label_stem(labels)}_mask.png"

    if output_path:
        path = Path(clean_path_arg(output_path))
        if path.suffix:
            return avoid_overwrite_path(path)
        return avoid_overwrite_path(path / filename)

    return avoid_overwrite_path(Path(default_dir) / filename)
