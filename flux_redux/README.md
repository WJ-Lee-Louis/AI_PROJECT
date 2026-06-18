# FLUX Redux

Modal-based inference scaffold for `black-forest-labs/FLUX.1-Redux-dev` with `black-forest-labs/FLUX.1-dev`.

Redux converts a reference image into FLUX conditioning embeddings:

```text
reference image -> FluxPriorReduxPipeline -> prompt_embeds / pooled_prompt_embeds -> FluxPipeline
```

This makes Redux useful for image variation and embedding-level research. Text prompt support exists through the Redux prior, but the default behavior is more image-prior driven than prompt-driven.

## Files

```text
flux_redux/
  modal_inference.py   # Modal remote function and local CLI entrypoint
  config.py            # model ids and default paths
  model_loader.py      # Redux prior + base FLUX loading
  conditioning.py      # reference preprocessing and embedding-scale controls
  inference.py         # generation flow
  LICENSE_NOTES.md
```

## Hugging Face access

`FLUX.1-dev` and `FLUX.1-Redux-dev` require Hugging Face access and license review. Use the same Modal Secret as other FLUX experiments:

```powershell
python -m modal secret create huggingface-secret HF_TOKEN=hf_your_token_here --force
```

## Run image-only Redux inference

```powershell
$env:PYTHONUTF8='1'
python -m modal run flux_redux/modal_inference.py `
  --reference-image-path "dreambooth_reference_dataset/dog/00.jpg" `
  --prompt ""
```

## Run with a text prompt

```powershell
python -m modal run flux_redux/modal_inference.py `
  --reference-image-path "dreambooth_reference_dataset/dog/00.jpg" `
  --prompt "a dog running on the beach"
```

## Useful controls

```text
--prompt-embeds-scale 1.0
--pooled-prompt-embeds-scale 1.0
--guidance-scale 2.5
--num-inference-steps 50
```

Outputs are written to:

```text
outputs/flux_redux/<reference_name>_generated.png
```

If the filename already exists, `_001`, `_002`, and so on are appended.
