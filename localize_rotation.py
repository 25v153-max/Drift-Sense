import cv2
import numpy as np
import json

# ============================================================
# DRIFT-SENSE
# STEP 8 - ROTATION-AWARE LOCALIZATION
# ============================================================

REFERENCE_PATH = "data/reference_001.png"
SEARCH_PATH = "data/search_001.png"
METADATA_PATH = "data/metadata_001.json"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET_SIZE = 100

# Test multiple possible rotations
ROTATION_ANGLES = np.arange(
    -4.0,
    4.01,
    0.5
)


# ------------------------------------------------------------
# 1. Load images
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
# 2. Resize reference
# ------------------------------------------------------------

reference_small = cv2.resize(
    reference,
    (TARGET_SIZE, TARGET_SIZE),
    interpolation=cv2.INTER_AREA
)


# ------------------------------------------------------------
# 3. Function to rotate template
# ------------------------------------------------------------

def rotate_image(image, angle):

    height, width = image.shape

    center = (
        width // 2,
        height // 2
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
# 4. Search across rotation angles
# ------------------------------------------------------------

best_score = -1

best_angle = None

best_location = None


for angle in ROTATION_ANGLES:

    rotated_template = rotate_image(
        reference_small,
        angle
    )


    result = cv2.matchTemplate(
        search,
        rotated_template,
        cv2.TM_CCOEFF_NORMED
    )


    min_value, max_value, min_location, max_location = (
        cv2.minMaxLoc(result)
    )


    if max_value > best_score:

        best_score = max_value

        best_angle = angle

        best_location = max_location


# ------------------------------------------------------------
# 5. Calculate predicted centre
# ------------------------------------------------------------

top_left_x = best_location[0]
top_left_y = best_location[1]


predicted_x = (
    top_left_x
    + TARGET_SIZE // 2
)

predicted_y = (
    top_left_y
    + TARGET_SIZE // 2
)


# ------------------------------------------------------------
# 6. Load ground truth
# ------------------------------------------------------------

with open(
    METADATA_PATH,
    "r"
) as file:

    metadata = json.load(file)


true_x = metadata["ground_truth"]["x"]
true_y = metadata["ground_truth"]["y"]


# ------------------------------------------------------------
# 7. Calculate localization error
# ------------------------------------------------------------

error = np.sqrt(
    (predicted_x - true_x) ** 2
    +
    (predicted_y - true_y) ** 2
)


# ------------------------------------------------------------
# 8. Print results
# ------------------------------------------------------------

print("=" * 65)

print(
    "DRIFT-SENSE - ROTATION-AWARE LOCALIZATION"
)

print("=" * 65)


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


print("\nEstimated rotation:")

print(
    f"{best_angle:.2f} degrees"
)


print("\nTemplate matching confidence:")

print(
    f"{best_score:.4f}"
)


print("\nLocalization error:")

print(
    f"{error:.2f} pixels"
)


print("\nRotation search range:")

print(
    "-4° to +4°"
)

print(
    "Step = 0.5°"
)


print("=" * 65)