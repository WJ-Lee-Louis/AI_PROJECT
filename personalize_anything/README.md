# Personalize Anything Single-Subject Modal Setup

This folder is a minimal Modal setup for running the single-subject path from
Personalize Anything. It intentionally excludes multi-subject composition and
inpainting/outpainting entrypoints.

## Inputs

The inference path uses:

- reference image
- binary subject mask
- reference prompt, used for the source/reference branch
- target prompt, used for the final generated image
- empty inversion prompt by default, matching the upstream notebook

White pixels in the mask are preserved/injected as reference subject tokens.
Black pixels are left freer for the target prompt and denoising process.

## 1. Generate A Mask With Grounded SAM

```powershell
python -m modal run personalize_anything/modal_mask.py `
  --reference-image-path "path/to/reference.jpg" `
  --labels "person"
```

The mask is saved under:

```text
personalize_anything/masks/
```

Useful options:

```powershell
--labels "person"
--threshold 0.25
--max-detections 1
```

Use simple detection labels such as `person`, `man`, `dog`, `cat`, or
`teddy bear`. Grounded SAM labels are segmentation targets, not full
generation prompts.

## 2. Run Single-Subject Personalization

```powershell
python -m modal run personalize_anything/modal_inference.py `
  --reference-image-path "path/to/reference.jpg" `
  --mask-path "personalize_anything/masks/reference_person_mask.png" `
  --reference-prompt "A person" `
  --prompt "A person eating a slice of pizza"
```

Outputs are saved under:

```text
outputs/personalize_anything/
```

If the target file already exists, the script appends `_01`, `_02`, and so on
to avoid overwriting previous results.

## Mask Editing

For identity-behavior disentanglement, keep identity-heavy regions white and
remove pose/behavior-heavy regions from the mask.

For a person:

- keep face, hair, and stable clothing cues white
- remove or reduce arms, hands, legs, and pose-defining silhouette regions

Recommended mask variants for each reference image:

- `mask_full_subject.png`
- `mask_identity_core.png`

You can create the first mask with Grounded SAM, then edit it in a UI such as:

- Meta Segment Anything demo: https://segment-anything.metademolab.com/
- Hugging Face Segment Anything Web: https://huggingface.co/spaces/Xenova/segment-anything-web
- GIMP, Krita, Paint.NET, Photoshop, or Photopea for direct black/white mask editing

Save the final mask as a PNG with white foreground and black background.

## Convert UI Cut-Outs To Binary Masks

Some SAM UIs download a cut-out or overlay image instead of a binary mask. If
the background is black, convert it with:

```powershell
python personalize_anything/convert_mask.py `
  --input "C:\Users\82107\Downloads\image.png" `
  --output "personalize_anything/masks/white_cat_face_binary_mask.png"
```

If the foreground has small black holes that should be preserved as subject
area, add:

```powershell
--fill-holes
```

## Important Caveat

Personalize Anything is not a reference-image-only model. It is a
mask-guided inversion/token-replacement method. Identity preservation should be
interpreted under the condition that a subject mask was supplied.
