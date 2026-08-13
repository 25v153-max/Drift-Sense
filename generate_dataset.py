import numpy as np
from PIL import Image
import os
import json

# ============================================================
# DRIFT-SENSE
# STEP 7 - Synthetic DRAM Dataset Generator
#
# Features:
#   - 1000 x 1000 reference image
#   - 1000 x 1000 search image
#   - 10:1 scale relationship
#   - Random target position
#   - Target rotation
#   - Gaussian noise
#   - Ground-truth coordinates
#   - Complete metadata
#   - Debug image
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000
TARGET_SIZE = 105

RANDOM_SEED = 42

# Target rotation in degrees
ROTATION_DEGREES = 3.0

# Gaussian noise strength
NOISE_SIGMA = 10.0


# Create reproducible random generator
rng = np.random.default_rng(RANDOM_SEED)


# Create output directory
os.makedirs("data", exist_ok=True)


# ============================================================
# 2. GENERATE DRAM-STYLE PATTERN
# ============================================================

def generate_dram_pattern(size):

    image = np.zeros((size, size), dtype=np.uint8)

    # --------------------------------------------------------
    # Dark background
    # --------------------------------------------------------

    image[:] = 35


    # --------------------------------------------------------
    # DRAM structural parameters
    # --------------------------------------------------------

    spacing = max(10, size // 10)

    line_width = max(1, size // 200)


    # --------------------------------------------------------
    # Vertical bit lines
    # --------------------------------------------------------

    for x in range(
        spacing // 2,
        size,
        spacing
    ):

        image[
            :,
            x:x + line_width
        ] = 170


    # --------------------------------------------------------
    # Horizontal word lines
    # --------------------------------------------------------

    for y in range(
        spacing // 2,
        size,
        spacing
    ):

        image[
            y:y + line_width,
            :
        ] = 170


    # --------------------------------------------------------
    # Contact / via dots
    # --------------------------------------------------------

    radius = max(
        2,
        size // 70
    )


    for x in range(
        spacing // 2,
        size,
        spacing
    ):

        for y in range(
            spacing // 2,
            size,
            spacing
        ):

            y1 = max(
                0,
                y - radius
            )

            y2 = min(
                size,
                y + radius + 1
            )

            x1 = max(
                0,
                x - radius
            )

            x2 = min(
                size,
                x + radius + 1
            )


            image[
                y1:y2,
                x1:x2
            ] = 245


    return image


# ============================================================
# 3. CREATE REFERENCE IMAGE
# ============================================================

reference = generate_dram_pattern(
    IMAGE_SIZE
)


reference_path = (
    "data/reference_001.png"
)


Image.fromarray(
    reference
).save(
    reference_path
)


# ============================================================
# 4. CREATE SEARCH IMAGE
# ============================================================

search = generate_dram_pattern(
    IMAGE_SIZE
)


# ============================================================
# 5. RESIZE REFERENCE TO SEARCH SCALE
#
# Reference represents 100×
# Search represents 10×
#
# Approximate scale ratio = 10:1
# ============================================================

reference_small = Image.fromarray(
    reference
).resize(
    (
        TARGET_SIZE,
        TARGET_SIZE
    ),
    Image.Resampling.LANCZOS
)


# ============================================================
# 6. ROTATE TARGET
# ============================================================

reference_small = reference_small.rotate(
    ROTATION_DEGREES,
    resample=Image.Resampling.BICUBIC,
    expand=False
)


reference_small = np.array(
    reference_small
)


# ============================================================
# 7. GENERATE RANDOM TARGET POSITION
# ============================================================

margin = (
    TARGET_SIZE // 2
    + 10
)


target_x = int(
    rng.integers(
        margin,
        IMAGE_SIZE - margin
    )
)


target_y = int(
    rng.integers(
        margin,
        IMAGE_SIZE - margin
    )
)


# ============================================================
# 8. CALCULATE TARGET BOUNDARIES
# ============================================================

x_start = (
    target_x
    - TARGET_SIZE // 2
)

y_start = (
    target_y
    - TARGET_SIZE // 2
)


x_end = (
    x_start
    + TARGET_SIZE
)

y_end = (
    y_start
    + TARGET_SIZE
)


# ============================================================
# 9. INSERT TARGET INTO SEARCH IMAGE
# ============================================================

search[
    y_start:y_end,
    x_start:x_end
] = reference_small


# ============================================================
# 10. ADD GAUSSIAN NOISE
# ============================================================

noise = rng.normal(
    loc=0,
    scale=NOISE_SIGMA,
    size=search.shape
)


noisy_search = (
    search.astype(np.float32)
    + noise
)


# ------------------------------------------------------------
# Keep pixel values within valid grayscale range
# ------------------------------------------------------------

noisy_search = np.clip(
    noisy_search,
    0,
    255
)


noisy_search = (
    noisy_search.astype(np.uint8)
)


# ============================================================
# 11. SAVE SEARCH IMAGE
# ============================================================

search_path = (
    "data/search_001.png"
)


Image.fromarray(
    noisy_search
).save(
    search_path
)


# ============================================================
# 12. CREATE DEBUG IMAGE
#
# The white marker shows the TRUE target centre.
# This is ONLY for debugging.
# Do NOT use this image for localization.
# ============================================================

debug_search = (
    noisy_search.copy()
)


marker_size = 8


debug_search[
    target_y - marker_size:
    target_y + marker_size,

    target_x - marker_size:
    target_x + marker_size
] = 255


debug_path = (
    "data/search_001_debug.png"
)


Image.fromarray(
    debug_search
).save(
    debug_path
)


# ============================================================
# 13. CREATE COMPLETE METADATA
# ============================================================

metadata = {

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    "dataset_name": "Drift-Sense Synthetic Dataset",

    "dataset_version": "0.1",

    "case_id": "001",


    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    "architecture": "DRAM",


    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    "reference_image": "reference_001.png",

    "search_image": "search_001.png",

    "image_type": "grayscale",

    "reference_image_size": [
        IMAGE_SIZE,
        IMAGE_SIZE
    ],

    "search_image_size": [
        IMAGE_SIZE,
        IMAGE_SIZE
    ],


    # --------------------------------------------------------
    # Magnification / scale
    # --------------------------------------------------------

    "reference_magnification": "100x",

    "search_magnification": "10x",

    "nominal_scale_ratio": 10.0,


    # --------------------------------------------------------
    # Target information
    # --------------------------------------------------------

   "nominal_target_size_in_search_pixels": [
    100,
    100
],

"actual_target_size_in_search_pixels": [
    TARGET_SIZE,
    TARGET_SIZE
],

"scale_variation_percent": 5.0,

    "target_center": {
        "x": target_x,
        "y": target_y
    },


    "ground_truth": {
        "x": target_x,
        "y": target_y
    },


    # --------------------------------------------------------
    # Transformation information
    # --------------------------------------------------------

    "transformations": {

        "nominal_scale_ratio": 10.0,
"actual_scale_ratio": 1000 / TARGET_SIZE,

        "rotation_degrees":
            ROTATION_DEGREES

    },


    # --------------------------------------------------------
    # Noise information
    # --------------------------------------------------------

    "noise": {

        "type": "Gaussian",

        "sigma": NOISE_SIGMA,

        "mean": 0.0

    },


    # --------------------------------------------------------
    # Randomness / reproducibility
    # --------------------------------------------------------

    "random_seed": RANDOM_SEED,


    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    "debug_image":
        "search_001_debug.png",

    "debug_marker": True,


    # --------------------------------------------------------
    # Generation information
    # --------------------------------------------------------

    "generator": "generate_dataset.py",

    "generator_step": 7

}


# ============================================================
# 14. SAVE METADATA
# ============================================================

metadata_path = (
    "data/metadata_001.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("=" * 65)

print(
    "DRIFT-SENSE - SYNTHETIC DATA GENERATOR"
)

print("=" * 65)


print("\nArchitecture      : DRAM")

print(
    "Reference size    : "
    f"{IMAGE_SIZE} × {IMAGE_SIZE}"
)

print(
    "Search size       : "
    f"{IMAGE_SIZE} × {IMAGE_SIZE}"
)

print(
    "Reference         : 100×"
)

print(
    "Search            : 10×"
)

print(
    "Scale ratio       : 10:1"
)

print(
    "Rotation          : "
    f"{ROTATION_DEGREES}°"
)

print(
    "Noise type        : Gaussian"
)

print(
    "Noise sigma       : "
    f"{NOISE_SIGMA}"
)


print("\nGround-truth target centre:")

print(
    f"x = {target_x}"
)

print(
    f"y = {target_y}"
)


print(
    "\nRandom seed       : "
    f"{RANDOM_SEED}"
)


print("\nFiles created:")

print(
    reference_path
)

print(
    search_path
)

print(
    debug_path
)

print(
    metadata_path
)


print(
    "\nGeneration complete!"
)

print("=" * 65)