import cv2
import numpy as np
import json

# ============================================================
# DRIFT-SENSE - V4
# EXTENDED SCALE + ROTATION AWARE LOCALIZATION
# ============================================================

REFERENCE_PATH = "data/reference_001.png"
SEARCH_PATH = "data/search_001.png"
METADATA_PATH = "data/metadata_001.json"

# ------------------------------------------------------------
# V4 search ranges
# ------------------------------------------------------------

# Expanded from V3's 90-110
TARGET_SIZES = list(range(85, 121, 5))

# Expanded from V3's -4° to +4°
ROTATION_ANGLES = np.arange(
    -8.0,
    8.01,
    0.5
)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.40


# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

reference = cv2.imread(
    REFERENCE_PATH,
    cv2.IMREAD_GRAYSCALE
)

search = cv2.imread(
    SEARCH_PATH,
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    raise FileNotFoundError(
        "Reference image could not be loaded."
    )

if search is None:
    raise FileNotFoundError(
        "Search image could not be loaded."
    )


# ------------------------------------------------------------
# Rotate template
# ------------------------------------------------------------

def rotate_image(image, angle):

    height, width = image.shape

    center = (
        width / 2,
        height / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=35
    )

    return rotated


# ------------------------------------------------------------
# Best-match search
# ------------------------------------------------------------

best_score = -1.0
best_angle = None
best_size = None
best_location = None

for target_size in TARGET_SIZES:

    template = cv2.resize(
        reference,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_AREA
    )

    for angle in ROTATION_ANGLES:

        rotated_template = rotate_image(
            template,
            angle
        )

        result = cv2.matchTemplate(
            search,
            rotated_template,
            cv2.TM_CCOEFF_NORMED
        )

        (
            min_value,
            max_value,
            min_location,
            max_location
        ) = cv2.minMaxLoc(
            result
        )

        if max_value > best_score:

            best_score = max_value
            best_angle = angle
            best_size = target_size
            best_location = max_location


# ------------------------------------------------------------
# Predicted centre
# ------------------------------------------------------------

top_left_x = best_location[0]
top_left_y = best_location[1]

predicted_x = int(
    round(
        top_left_x
        + best_size / 2
    )
)

predicted_y = int(
    round(
        top_left_y
        + best_size / 2
    )
)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)


true_x = metadata["ground_truth"]["x"]
true_y = metadata["ground_truth"]["y"]


# ------------------------------------------------------------
# Localization error
# ------------------------------------------------------------

error = np.sqrt(
    (predicted_x - true_x) ** 2
    +
    (predicted_y - true_y) ** 2
)


# ------------------------------------------------------------
# Estimated scale
# ------------------------------------------------------------

estimated_scale_ratio = (
    1000 / best_size
)

scale_variation_percent = (
    (best_size - 100)
    / 100
) * 100


# ------------------------------------------------------------
# Confidence status
# ------------------------------------------------------------

if best_score >= CONFIDENCE_THRESHOLD:

    status = "RELIABLE"

else:

    status = "LOW CONFIDENCE"


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("=" * 70)

print(
    "DRIFT-SENSE - V4"
)

print(
    "EXTENDED SCALE + ROTATION AWARE LOCALIZATION"
)

print("=" * 70)


print("\nGround-truth centre:")

print(
    f"x = {true_x}"
)

print(
    f"y = {true_y}"
)


print("\nPredicted centre:")

print(
    f"x = {predicted_x}"
)

print(
    f"y = {predicted_y}"
)


print("\nEstimated target size:")

print(
    f"{best_size} × {best_size} pixels"
)


print("\nEstimated scale ratio:")

print(
    f"{estimated_scale_ratio:.4f}:1"
)


print("\nEstimated scale variation:")

print(
    f"{scale_variation_percent:+.2f}%"
)


print("\nEstimated rotation:")

print(
    f"{best_angle:.2f} degrees"
)


print("\nTemplate matching confidence:")

print(
    f"{best_score:.4f}"
)


print("\nLocalization status:")

print(
    status
)


print("\nLocalization error:")

print(
    f"{error:.2f} pixels"
)


print("\nScale search range:")

print(
    "85 to 120 pixels"
)


print("\nRotation search range:")

print(
    "-8° to +8°"
)

print(
    "Step = 0.5°"
)


print("=" * 70)