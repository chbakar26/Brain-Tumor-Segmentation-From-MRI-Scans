"""
Configuration template for Brain Tumor Segmentation App
Copy this file to 'config_local.py' and update the paths for your system.
"""

# Model Configuration
MODEL_PATH = "path/to/your/best_brats_model_dice.pth"
MODEL_DEF_PATH = "path/to/your/improved3dunet.py"
MODEL_CLASS = "Improved3DUNet"
BASE_FILTERS = 16

# Inference Configuration
ZRANGE = "60:100"
PATCH_SIZE = "128,128,64"
OVERLAP = 0.35
FORCE_CPU = True
INTENSITY_NORM = "zscore"
LOW_MEM_ACCUM = False
ACCUM_DTYPE = "float16"
AMP = "auto"
