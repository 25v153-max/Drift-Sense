import cv2
import numpy as np
import json

# ============================================================
# DRIFT-SENSE
# STEP 4 - First Scale-Aware Localization Baseline
# ============================================================

REFERENCE_PATH = "data/reference_001.png"
SEARCH_PATH = "data/search_001.png"
METADATA_PATH = "data/metadata_001.json"


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
# 2. Resize reference to the search-image scale
#
# Reference = 100×
# Search    = 10×
#
# Therefore approximately 10:1 scale difference.
# ------------------------------------------------------------

reference_small = cv2.resize(
    reference,
    (100, 100),
    interpolation=cv2.INTER_AREA
)


# ------------------------------------------------------------
# 3. Perform template matching
# ------------------------------------------------------------

result = cv2.matchTemplate(
    search,
    reference_small,
    cv2.TM_CCOEFF_NORMED
)


# ------------------------------------------------------------
# 4. Find the best matching location
# ------------------------------------------------------------

min_value, max_value, min_location, max_location = cv2.minMaxLoc(
    result
)


# max_location gives the TOP-LEFT corner of the match.

top_left_x = max_location[0]
top_left_y = max_location[1]


# ------------------------------------------------------------
# 5. Calculate centre coordinate
# ------------------------------------------------------------

template_width = reference_small.shape[1]
template_height = reference_small.shape[0]

predicted_x = top_left_x + template_width // 2
predicted_y = top_left_y + template_height // 2


# ------------------------------------------------------------
# 6. Load ground-truth coordinates
# ------------------------------------------------------------

with open(METADATA_PATH, "r") as file:

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

print("=" * 60)
print("DRIFT-SENSE - LOCALIZATION RESULT")
print("=" * 60)

print("\nGround-truth centre:")
print(f"x = {true_x}")
print(f"y = {true_y}")

print("\nPredicted centre:")
print(f"x = {predicted_x}")
print(f"y = {predicted_y}")

print("\nTemplate matching confidence:")
print(f"{max_value:.4f}")

print("\nLocalization error:")
print(f"{error:.2f} pixels")

print("\n" + "=" * 60)