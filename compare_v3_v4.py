import cv2
import numpy as np
import csv
import os

# ============================================================
# DRIFT-SENSE
# V3 vs V4 FAIR COMPARISON
#
# Both algorithms are tested on EXACTLY the same 40 cases.
#
# V3:
#   Scale    : 90 - 110
#   Rotation : -4° to +4°
#
# V4:
#   Scale    : 85 - 120
#   Rotation : -8° to +8°
#
# Success threshold:
#   Localization error <= 5 pixels
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000
NUM_CASES = 40

SUCCESS_THRESHOLD = 5.0

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "v3_v4_comparison.csv"
)


# ============================================================
# TEST PARAMETERS
# ============================================================

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

BLUR_KERNELS = [
    0,
    3,
    5
]


# ============================================================
# V3 SEARCH RANGE
# ============================================================

V3_SIZES = list(
    range(90, 111, 5)
)

V3_ROTATIONS = np.arange(
    -4.0,
    4.01,
    0.5
)


# ============================================================
# V4 SEARCH RANGE
# ============================================================

V4_SIZES = list(
    range(85, 121, 5)
)

V4_ROTATIONS = np.arange(
    -8.0,
    8.01,
    0.5
)


# ============================================================
# DRAM PATTERN
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

    # Contact points
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
# ROTATION
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
# GENERATE ONE TEST CASE
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

    # Base search image
    search = generate_dram_pattern(
        IMAGE_SIZE
    )

    # Resize target
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

    # Position
    x_start = (
        true_x
        - target_size // 2
    )

    y_start = (
        true_y
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

    # Gaussian noise
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

    # Gaussian blur
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
        true_x,
        true_y
    )


# ============================================================
# LOCALIZATION FUNCTION
# ============================================================

def localize(
    reference,
    search,
    candidate_sizes,
    candidate_rotations
):

    best_score = -1.0
    best_size = None
    best_angle = None
    best_location = None

    for target_size in candidate_sizes:

        template = cv2.resize(
            reference,
            (
                target_size,
                target_size
            ),
            interpolation=cv2.INTER_AREA
        )

        for angle in candidate_rotations:

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

                best_size = target_size

                best_angle = angle

                best_location = max_location

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
# CREATE REFERENCE
# ============================================================

print("=" * 72)

print(
    "DRIFT-SENSE - V3 vs V4 FAIR COMPARISON"
)

print("=" * 72)

print(
    "\nGenerating identical test cases for V3 and V4..."
)

print(
    f"Total cases : {NUM_CASES}"
)

print(
    "\nV3:"
)

print(
    "Scale = 90–110"
)

print(
    "Rotation = -4° to +4°"
)

print(
    "\nV4:"
)

print(
    "Scale = 85–120"
)

print(
    "Rotation = -8° to +8°"
)

print()


reference = generate_dram_pattern(
    IMAGE_SIZE
)


# ============================================================
# CSV
# ============================================================

fieldnames = [

    "case_id",

    "target_size",
    "rotation",
    "noise",
    "blur",

    "ground_truth_x",
    "ground_truth_y",

    "v3_x",
    "v3_y",
    "v3_error",
    "v3_size",
    "v3_rotation",
    "v3_confidence",
    "v3_success",

    "v4_x",
    "v4_y",
    "v4_error",
    "v4_size",
    "v4_rotation",
    "v4_confidence",
    "v4_success"

]


results = []


# ============================================================
# FIXED RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    3030
)


# ============================================================
# GENERATE AND TEST 40 IDENTICAL CASES
# ============================================================

for case_id in range(
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
        + case_id
    )

    # --------------------------------------------------------
    # Generate ONE image
    # --------------------------------------------------------

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


    # ========================================================
    # V3
    # ========================================================

    (
        v3_x,
        v3_y,
        v3_size,
        v3_rotation,
        v3_confidence
    ) = localize(
        reference,
        search,
        V3_SIZES,
        V3_ROTATIONS
    )

    v3_error = np.sqrt(
        (v3_x - true_x) ** 2
        +
        (v3_y - true_y) ** 2
    )

    v3_success = (
        v3_error
        <= SUCCESS_THRESHOLD
    )


    # ========================================================
    # V4
    # ========================================================

    (
        v4_x,
        v4_y,
        v4_size,
        v4_rotation,
        v4_confidence
    ) = localize(
        reference,
        search,
        V4_SIZES,
        V4_ROTATIONS
    )

    v4_error = np.sqrt(
        (v4_x - true_x) ** 2
        +
        (v4_y - true_y) ** 2
    )

    v4_success = (
        v4_error
        <= SUCCESS_THRESHOLD
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "case_id": case_id,

        "target_size": target_size,

        "rotation": rotation,

        "noise": noise_sigma,

        "blur": blur_kernel,

        "ground_truth_x": true_x,

        "ground_truth_y": true_y,

        "v3_x": v3_x,

        "v3_y": v3_y,

        "v3_error": round(
            v3_error,
            2
        ),

        "v3_size": v3_size,

        "v3_rotation": v3_rotation,

        "v3_confidence": round(
            v3_confidence,
            4
        ),

        "v3_success": v3_success,

        "v4_x": v4_x,

        "v4_y": v4_y,

        "v4_error": round(
            v4_error,
            2
        ),

        "v4_size": v4_size,

        "v4_rotation": v4_rotation,

        "v4_confidence": round(
            v4_confidence,
            4
        ),

        "v4_success": v4_success

    }

    results.append(
        result
    )


    # ========================================================
    # PRINT CASE
    # ========================================================

    v3_status = (
        "PASS"
        if v3_success
        else "FAIL"
    )

    v4_status = (
        "PASS"
        if v4_success
        else "FAIL"
    )

    print(
        f"Case {case_id:02d} | "
        f"Size {target_size:3d} | "
        f"Rot {rotation:+.0f}° | "
        f"Noise {noise_sigma:2.0f} | "
        f"Blur {blur_kernel} | "
        f"V3 {v3_error:6.2f}px {v3_status} | "
        f"V4 {v4_error:6.2f}px {v4_status}"
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_PATH,
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
# STATISTICS FUNCTION
# ============================================================

def calculate_statistics(
    algorithm
):

    errors = np.array([
        result[
            f"{algorithm}_error"
        ]
        for result in results
    ])

    confidences = np.array([
        result[
            f"{algorithm}_confidence"
        ]
        for result in results
    ])

    successes = np.array([
        result[
            f"{algorithm}_success"
        ]
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

    return {

        "successful": successful_cases,

        "failed": failed_cases,

        "success_rate": success_rate,

        "mean_error": np.mean(
            errors
        ),

        "median_error": np.median(
            errors
        ),

        "min_error": np.min(
            errors
        ),

        "max_error": np.max(
            errors
        ),

        "mean_confidence":
            np.mean(
                confidences
            )

    }


# ============================================================
# CALCULATE V3 / V4 STATISTICS
# ============================================================

v3_stats = calculate_statistics(
    "v3"
)

v4_stats = calculate_statistics(
    "v4"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")

print("=" * 72)

print(
    "FINAL V3 vs V4 COMPARISON"
)

print("=" * 72)


print("\nV3 RESULTS")

print("-" * 40)

print(
    f"Successful cases : "
    f"{v3_stats['successful']}"
)

print(
    f"Failed cases     : "
    f"{v3_stats['failed']}"
)

print(
    f"Success rate     : "
    f"{v3_stats['success_rate']:.2f}%"
)

print(
    f"Mean error       : "
    f"{v3_stats['mean_error']:.2f} pixels"
)

print(
    f"Median error     : "
    f"{v3_stats['median_error']:.2f} pixels"
)

print(
    f"Minimum error    : "
    f"{v3_stats['min_error']:.2f} pixels"
)

print(
    f"Maximum error    : "
    f"{v3_stats['max_error']:.2f} pixels"
)

print(
    f"Mean confidence  : "
    f"{v3_stats['mean_confidence']:.4f}"
)


print("\nV4 RESULTS")

print("-" * 40)

print(
    f"Successful cases : "
    f"{v4_stats['successful']}"
)

print(
    f"Failed cases     : "
    f"{v4_stats['failed']}"
)

print(
    f"Success rate     : "
    f"{v4_stats['success_rate']:.2f}%"
)

print(
    f"Mean error       : "
    f"{v4_stats['mean_error']:.2f} pixels"
)

print(
    f"Median error     : "
    f"{v4_stats['median_error']:.2f} pixels"
)

print(
    f"Minimum error    : "
    f"{v4_stats['min_error']:.2f} pixels"
)

print(
    f"Maximum error    : "
    f"{v4_stats['max_error']:.2f} pixels"
)

print(
    f"Mean confidence  : "
    f"{v4_stats['mean_confidence']:.4f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

success_improvement = (
    v4_stats["success_rate"]
    - v3_stats["success_rate"]
)

error_reduction = (
    v3_stats["mean_error"]
    - v4_stats["mean_error"]
)

print("\n")

print("=" * 72)

print(
    "V4 IMPROVEMENT"
)

print("=" * 72)

print(
    f"\nSuccess-rate improvement : "
    f"{success_improvement:+.2f} percentage points"
)

print(
    f"Mean-error reduction     : "
    f"{error_reduction:+.2f} pixels"
)


# ============================================================
# WINNER
# ============================================================

if (
    v4_stats["success_rate"]
    > v3_stats["success_rate"]
):

    print(
        "\nOverall result : V4 IMPROVED"
    )

elif (
    v4_stats["success_rate"]
    < v3_stats["success_rate"]
):

    print(
        "\nOverall result : V3 BETTER"
    )

else:

    print(
        "\nOverall result : SAME SUCCESS RATE"
    )


print("\nResults saved to:")

print(
    CSV_PATH
)


print("\n")

print("=" * 72)

print(
    "V3 vs V4 COMPARISON COMPLETE!"
)

print("=" * 72)