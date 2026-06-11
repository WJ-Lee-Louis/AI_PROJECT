APP_NAME = "xlabs-flux-ip-adapter-inference"

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
IP_ADAPTER_ID = "XLabs-AI/flux-ip-adapter"
IP_ADAPTER_WEIGHT_NAME = "ip_adapter.safetensors"
IMAGE_ENCODER_ID = "openai/clip-vit-large-patch14"

DEFAULT_REFERENCE_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "diffusers/flux_ip_adapter_input.jpg"
)
DEFAULT_REFERENCE_IMAGE_PATH = ""
DEFAULT_OUTPUT_DIR = "outputs/xlabs_flux_ip_adapter"
DEFAULT_OUTPUT_PATH = ""
