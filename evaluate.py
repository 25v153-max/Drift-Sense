import cv2
import numpy as np
import json
import csv
import os

# ============================================================
# DRIFT-SENSE
# STEP 11 - AUTOMATED 30-CASE EVALUATION
# ============================================================

IMAGE_SIZE = 1000
REFERENCE_SIZE = 1000

# Number of independent test cases
NUM_CASES = 30

# Candidate target sizes
TARGET_SIZES = [90, 95, 100, 105, 110]

# Candidate rotations
ROTATIONS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

# Candidate noise levels
NOISE_LEVELS = [0, 5, 10, 15, 20]

# V3 search configuration
SEARCH_SIZES = [90, 95, 100, 105, 110]

SEARCH_ROTATIONS = np.arange(
    -4.0,
    4.01,
    0.5
)

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
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
# 3. Generate one test case
# ============================================================

def generate_case(
    reference,
    target_size,
    rotation,
    noise_sigma,
    seed
):

    rng = np.random.default_rng(seed)

    search = generate_dram_pattern(
        IMAGE_SIZE
    )

    # Resize reference
    target = cv2.resize(
        reference,
        (
            target_size,
            target_size
        ),
        interpolation=cv2.INTER_AREA
    )

    # Rotate target
    target = rotate_image(
        target,
        rotation
    )

    # Safe random target position
    margin = (
        target_size // 2
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

    # Target boundaries
    x_start = (
        target_x
        - target_size // 2
    )

    y_start = (
        target_y
        - target_size // 2
    )

    x_end = (
        x_start
        + target_size
    )

    y_end = (
        y_start
        + target_size
    )

    # Insert target
    search[
        y_start:y_end,
        x_start:x_end
    ] = target

    # Add Gaussian noise
    if noise_sigma > 0:

        noise = rng.normal(
            0,
            noise_sigma,
            search.shape
        )

        search = (
            search.astype(np.float32)
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

    return (
        search,
        target_x,
        target_y
    )


# ============================================================
# 4. V3 localization
# ============================================================

def localize(
    reference,
    search
):

    best_score = -1.0

    best_angle = None

    best_size = None

    best_location = None

    for target_size in SEARCH_SIZES:

        template = cv2.resize(
            reference,
            (
                target_size,
                target_size
            ),
            interpolation=cv2.INTER_AREA
        )

        for angle in SEARCH_ROTATIONS:

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

    # Predicted centre
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

    return (
        predicted_x,
        predicted_y,
        best_size,
        best_angle,
        best_score
    )


# ============================================================
# 5. Create reference
# ============================================================

print("=" * 70)

print(
    "DRIFT-SENSE - 30 CASE EVALUATION"
)

print("=" * 70)

print(
    f"\nGenerating {NUM_CASES} independent test cases..."
)


reference = generate_dram_pattern(
    REFERENCE_SIZE
)


# ============================================================
# 6. CSV result file
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "evaluation_results.csv"
)


fieldnames = [

    "case_id",

    "target_size",

    "rotation",

    "noise_sigma",

    "ground_truth_x",

    "ground_truth_y",

    "predicted_x",

    "predicted_y",

    "error_pixels",

    "estimated_size",

    "estimated_rotation",

    "confidence",

    "success"

]


results = []


# ============================================================
# 7. Run 30 independent cases
# ============================================================

rng = np.random.default_rng(
    2026
)


for case_number in range(
    1,
    NUM_CASES + 1
):

    target_size = int(
        rng.choice(
            TARGET_SIZES
        )
    )

    rotation = float(
        rng.choice(
            ROTATIONS
        )
    )

    noise_sigma = float(
        rng.choice(
            NOISE_LEVELS
        )
    )

    seed = 1000 + case_number

    # Generate case
    (
        search,
        true_x,
        true_y
    ) = generate_case(
        reference,
        target_size,
        rotation,
        noise_sigma,
        seed
    )

    # Localize
    (
        predicted_x,
        predicted_y,
        estimated_size,
        estimated_rotation,
        confidence
    ) = localize(
        reference,
        search
    )

    # Calculate error
    error = np.sqrt(
        (predicted_x - true_x) ** 2
        +
        (predicted_y - true_y) ** 2
    )

    # Success threshold
    success = error <= 5.0

    result = {

        "case_id": case_number,

        "target_size": target_size,

        "rotation": rotation,

        "noise_sigma": noise_sigma,

        "ground_truth_x": true_x,

        "ground_truth_y": true_y,

        "predicted_x": predicted_x,

        "predicted_y": predicted_y,

        "error_pixels": round(
            error,
            2
        ),

        "estimated_size": estimated_size,

        "estimated_rotation": estimated_rotation,

        "confidence": round(
            confidence,
            4
        ),

        "success": success

    }

    results.append(
        result
    )

    print(
        f"Case {case_number:02d} | "
        f"Size {target_size} | "
        f"Rotation {rotation:+.0f}° | "
        f"Noise {noise_sigma:.0f} | "
        f"Error {error:.2f}px | "
        f"Confidence {confidence:.4f} | "
        f"{'PASS' if success else 'FAIL'}"
    )


# ============================================================
# 8. Save CSV
# ============================================================

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 9. Calculate statistics
# ============================================================

errors = np.array([
    r["error_pixels"]
    for r in results
])

confidences = np.array([
    r["confidence"]
    for r in results
])

successes = np.array([
    r["success"]
    for r in results
])


mean_error = np.mean(
    errors
)

median_error = np.median(
    errors
)

max_error = np.max(
    errors
)

min_error = np.min(
    errors
)

mean_confidence = np.mean(
    confidences
)

success_rate = (
    np.sum(successes)
    / NUM_CASES
    * 100
)


# ============================================================
# 10. Print final evaluation
# ============================================================

print("\n")

print("=" * 70)

print(
    "FINAL EVALUATION"
)

print("=" * 70)


print(
    f"\nTotal cases      : {NUM_CASES}"
)

print(
    f"Successful cases : {np.sum(successes)}"
)

print(
    f"Failed cases     : "
    f"{NUM_CASES - np.sum(successes)}"
)

print(
    f"Success rate     : "
    f"{success_rate:.2f}%"
)

print(
    f"\nMean error       : "
    f"{mean_error:.2f} pixels"
)

print(
    f"Median error     : "
    f"{median_error:.2f} pixels"
)

print(
    f"Minimum error    : "
    f"{min_error:.2f} pixels"
)

print(
    f"Maximum error    : "
    f"{max_error:.2f} pixels"
)

print(
    f"\nMean confidence  : "
    f"{mean_confidence:.4f}"
)


print(
    "\nResults saved to:"
)

print(
    csv_path
)


print("\n")

print("=" * 70)

print(
    "30-CASE EVALUATION COMPLETE!"
)

print("=" * 70)