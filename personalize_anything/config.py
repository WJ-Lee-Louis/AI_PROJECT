APP_NAME = "personalize-anything-inference"
MASK_APP_NAME = "personalize-anything-mask"
MASK_GPU = "T4"

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"

DEFAULT_REFERENCE_IMAGE_PATH = "personalize_anything_example_data/teddy_bear/background.png"
DEFAULT_MASK_PATH = "personalize_anything_example_data/teddy_bear/mask.png"
DEFAULT_SEGMENTATION_LABELS = "teddy bear"

DEFAULT_REFERENCE_PROMPT = "A teddy bear"
DEFAULT_PROMPT = "A teddy bear running on the beach"
DEFAULT_INVERSION_PROMPT = ""

DEFAULT_OUTPUT_DIR = "outputs/personalize_anything"
DEFAULT_OUTPUT_PATH = ""

DEFAULT_MASK_OUTPUT_DIR = "personalize_anything/masks"
DEFAULT_MASK_OUTPUT_PATH = ""
