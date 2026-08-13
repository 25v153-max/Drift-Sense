import cv2
import numpy as np
import csv
import os

# ============================================================
# DRIFT-SENSE
# STEP 12 - V3 STRESS TEST
#
# Tests V3 beyond its original operating range:
#   Target size : 85 - 120 pixels
#   Rotation    : -8° to +8°
#   Noise       : 0 - 35
#   Blur        : Gaussian blur
#
# V3 itself is NOT modified.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000
REFERENCE_SIZE = 1000

NUM_CASES = 40

# Wider than the normal V3 evaluation range
TARGET_SIZES = [
    85,
    90,
    95,
    100,
    105,
    110,
    115,
    120
]

ROTATIONS = [
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    6,
    8
]

NOISE_LEVELS = [
    0,
    10,
    20,
    30,
    35
]

# Gaussian blur kernel sizes
BLUR_KERNELS = [
    0,
    3,
    5
]

# V3 search range
SEARCH_SIZES = [
    90,
    95,
    100,
    105,
    110
]

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
# 2. DRAM PATTERN
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
# 3. ROTATION FUNCTION
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
# 4. GENERATE STRESS TEST CASE
# ============================================================

def generate_case(
    reference,
    target_size,
    rotation,
    noise_sigma,
    blur_kernel,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    # Create search image
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

    # Random target centre
    margin = (
        target_size // 2
        + 20
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

    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------

    if noise_sigma > 0:

        noise = rng.normal(
            0,
            noise_sigma,
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

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------

    if blur_kernel > 0:

        search = cv2.GaussianBlur(
            search,
            (
                blur_kernel,
                blur_kernel
            ),
            0
        )

    return (
        search,
        target_x,
        target_y
    )


# ============================================================
# 5. V3 LOCALIZATION
# ============================================================

def localize_v3(
    reference,
    search
):

    best_score = -1.0
    best_angle = None
    best_size = None
    best_location = None

    # Search candidate scales
    for target_size in SEARCH_SIZES:

        template = cv2.resize(
            reference,
            (
                target_size,
                target_size
            ),
            interpolation=cv2.INTER_AREA
        )

        # Search candidate rotations
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
# 6. CREATE REFERENCE
# ============================================================

print("=" * 72)

print(
    "DRIFT-SENSE - V3 STRESS TEST"
)

print("=" * 72)

print(
    f"\nGenerating {NUM_CASES} stress-test cases..."
)

print(
    "Size range     : 85 - 120 pixels"
)

print(
    "Rotation range : -8° to +8°"
)

print(
    "Noise range    : 0 - 35"
)

print(
    "Blur           : 0, 3, 5"
)

print()


reference = generate_dram_pattern(
    REFERENCE_SIZE
)


# ============================================================
# 7. CSV SETUP
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "stress_test_results.csv"
)

fieldnames = [

    "case_id",

    "target_size",

    "rotation",

    "noise_sigma",

    "blur_kernel",

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
# 8. RANDOM CASE GENERATION
# ============================================================

rng = np.random.default_rng(
    3030
)


# ============================================================
# 9. RUN STRESS TEST
# ============================================================

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

    blur_kernel = int(
        rng.choice(
            BLUR_KERNELS
        )
    )

    seed = (
        5000
        + case_number
    )

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
        blur_kernel,
        seed
    )

    # V3 localization
    (
        predicted_x,
        predicted_y,
        estimated_size,
        estimated_rotation,
        confidence
    ) = localize_v3(
        reference,
        search
    )

    # Localization error
    error = np.sqrt(
        (predicted_x - true_x) ** 2
        +
        (predicted_y - true_y) ** 2
    )

    # Success threshold
    success = (
        error <= 5.0
    )

    result = {

        "case_id": case_number,

        "target_size": target_size,

        "rotation": rotation,

        "noise_sigma": noise_sigma,

        "blur_kernel": blur_kernel,

        "ground_truth_x": true_x,

        "ground_truth_y": true_y,

        "predicted_x": predicted_x,

        "predicted_y": predicted_y,

        "error_pixels": round(
            error,
            2
        ),

        "estimated_size": estimated_size,

        "estimated_rotation":
            estimated_rotation,

        "confidence": round(
            confidence,
            4
        ),

        "success": success

    }

    results.append(
        result
    )

    status = (
        "PASS"
        if success
        else "FAIL"
    )

    print(
        f"Case {case_number:02d} | "
        f"Size {target_size:3d} | "
        f"Rot {rotation:+.0f}° | "
        f"Noise {noise_sigma:2.0f} | "
        f"Blur {blur_kernel} | "
        f"Error {error:6.2f}px | "
        f"Conf {confidence:.4f} | "
        f"{status}"
    )


# ============================================================
# 10. SAVE CSV
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
# 11. STATISTICS
# ============================================================

errors = np.array([
    result["error_pixels"]
    for result in results
])

confidences = np.array([
    result["confidence"]
    for result in results
])

successes = np.array([
    result["success"]
    for result in results
])


successful_cases = int(
    np.sum(successes)
)

failed_cases = (
    NUM_CASES
    - successful_cases
)

success_rate = (
    successful_cases
    / NUM_CASES
    * 100
)

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


# ============================================================
# 12. FIND WORST CASES
# ============================================================

sorted_results = sorted(
    results,
    key=lambda x: x["error_pixels"],
    reverse=True
)


worst_cases = sorted_results[:5]


# ============================================================
# 13. FINAL REPORT
# ============================================================

print("\n")

print("=" * 72)

print(
    "STRESS TEST RESULTS"
)

print("=" * 72)


print(
    f"\nTotal cases       : "
    f"{NUM_CASES}"
)

print(
    f"Successful cases  : "
    f"{successful_cases}"
)

print(
    f"Failed cases      : "
    f"{failed_cases}"
)

print(
    f"Success rate      : "
    f"{success_rate:.2f}%"
)

print(
    f"\nMean error        : "
    f"{mean_error:.2f} pixels"
)

print(
    f"Median error      : "
    f"{median_error:.2f} pixels"
)

print(
    f"Minimum error     : "
    f"{min_error:.2f} pixels"
)

print(
    f"Maximum error     : "
    f"{max_error:.2f} pixels"
)

print(
    f"\nMean confidence   : "
    f"{mean_confidence:.4f}"
)


print("\nWorst 5 cases:")

print("-" * 72)


for result in worst_cases:

    print(
        f"Case {result['case_id']:02d} | "
        f"Size {result['target_size']} | "
        f"Rotation {result['rotation']:+.0f}° | "
        f"Noise {result['noise_sigma']:.0f} | "
        f"Blur {result['blur_kernel']} | "
        f"Error {result['error_pixels']:.2f}px | "
        f"Confidence {result['confidence']:.4f}"
    )


print("\nResults saved to:")

print(
    csv_path
)


print("\n")

print("=" * 72)

print(
    "STRESS TEST COMPLETE!"
)

print("=" * 72)