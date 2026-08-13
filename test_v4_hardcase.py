import cv2
import numpy as np

# ============================================================
# DRIFT-SENSE
# V4 HARD-CASE TEST
#
# Exact stress-test condition:
# Target size : 120 x 120
# Rotation    : +8 degrees
# Noise       : 0
# Blur        : 5
#
# This reproduces the type of case where V3 failed badly.
# ============================================================


IMAGE_SIZE = 1000

TARGET_SIZE = 120
ROTATION = 8.0
NOISE_SIGMA = 0
BLUR_KERNEL = 5

# Fixed seed for reproducibility
SEED = 5004

# V4 search range
SEARCH_SIZES = list(
    range(85, 121, 5)
)

SEARCH_ROTATIONS = np.arange(
    -8.0,
    8.01,
    0.5
)


# ============================================================
# 1. Generate DRAM pattern
# ============================================================

def generate_dram_pattern(size):

    image = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    image[:] = 35

    spacing = max(
        10,
        size // 10
    )

    line_width = max(
        1,
        size // 200
    )

    # Vertical lines
    for x in range(
        spacing // 2,
        size,
        spacing
    ):

        image[
            :,
            x:x + line_width
        ] = 170

    # Horizontal lines
    for y in range(
        spacing // 2,
        size,
        spacing
    ):

        image[
            y:y + line_width,
            :
        ] = 170

    # Contact dots
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
# 2. Rotate image
# ============================================================

def rotate_image(
    image,
    angle
):

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


# ============================================================
# 3. Generate hard-case image
# ============================================================

rng = np.random.default_rng(
    SEED
)

reference = generate_dram_pattern(
    IMAGE_SIZE
)

search = generate_dram_pattern(
    IMAGE_SIZE
)

# Resize reference to 120 x 120
target = cv2.resize(
    reference,
    (
        TARGET_SIZE,
        TARGET_SIZE
    ),
    interpolation=cv2.INTER_AREA
)

# Rotate target by +8 degrees
target = rotate_image(
    target,
    ROTATION
)

# ------------------------------------------------------------
# Random but reproducible target centre
# ------------------------------------------------------------

margin = (
    TARGET_SIZE // 2
    + 20
)

true_x = int(
    rng.integers(
        margin,
        IMAGE_SIZE - margin
    )
)

true_y = int(
    rng.integers(
        margin,
        IMAGE_SIZE - margin
    )
)

# Target placement
x_start = (
    true_x
    - TARGET_SIZE // 2
)

y_start = (
    true_y
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

search[
    y_start:y_end,
    x_start:x_end
] = target


# ============================================================
# 4. Add noise
# ============================================================

if NOISE_SIGMA > 0:

    noise = rng.normal(
        0,
        NOISE_SIGMA,
        search.shape
    )

    search = (
        search.astype(
            np.float32
        )
        + noise
    )

    search = np.clip(
        search,
        0,
        255
    )

    search = search.astype(
        np.uint8
    )


# ============================================================
# 5. Add blur
# ============================================================

if BLUR_KERNEL > 0:

    search = cv2.GaussianBlur(
        search,
        (
            BLUR_KERNEL,
            BLUR_KERNEL
        ),
        0
    )


# ============================================================
# 6. V4 localization
# ============================================================

best_score = -1.0
best_size = None
best_angle = None
best_location = None


for candidate_size in SEARCH_SIZES:

    template = cv2.resize(
        reference,
        (
            candidate_size,
            candidate_size
        ),
        interpolation=cv2.INTER_AREA
    )

    for candidate_angle in SEARCH_ROTATIONS:

        rotated_template = rotate_image(
            template,
            candidate_angle
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

            best_size = candidate_size

            best_angle = candidate_angle

            best_location = max_location


# ============================================================
# 7. Predicted centre
# ============================================================

predicted_x = int(
    round(
        best_location[0]
        + best_size / 2
    )
)

predicted_y = int(
    round(
        best_location[1]
        + best_size / 2
    )
)


# ============================================================
# 8. Localization error
# ============================================================

error = np.sqrt(
    (predicted_x - true_x) ** 2
    +
    (predicted_y - true_y) ** 2
)


# ============================================================
# 9. Confidence status
# ============================================================

if best_score >= 0.40:

    status = "RELIABLE"

else:

    status = "LOW CONFIDENCE"


# ============================================================
# 10. Print results
# ============================================================

print("=" * 70)

print(
    "DRIFT-SENSE - V4 HARD-CASE TEST"
)

print("=" * 70)


print("\nTest conditions:")

print(
    f"Target size : {TARGET_SIZE} × {TARGET_SIZE}"
)

print(
    f"Rotation    : {ROTATION:+.1f}°"
)

print(
    f"Noise       : {NOISE_SIGMA}"
)

print(
    f"Blur        : {BLUR_KERNEL} × {BLUR_KERNEL}"
)

print(
    f"Random seed : {SEED}"
)


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
    f"{best_size} × {best_size}"
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


print("\nV4 search range:")

print(
    "Scale: 85–120 pixels"
)

print(
    "Rotation: -8° to +8°"
)

print(
    "Rotation step: 0.5°"
)


print("=" * 70)