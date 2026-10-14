import json
import numpy as np
from PIL import ImageOps


def preprocess_image(image, target_size=(224, 224)):
    # 1. Convert to RGB
    image = image.convert("RGB")

    # 2. Get original dimensions
    original_width, original_height = image.size

    target_width, target_height = target_size

    # 3. Calculate scale while preserving aspect ratio
    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    # 4. Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 5. Resize
    image = image.resize((new_width, new_height))

    # 6. Calculate padding
    pad_width = target_width - new_width
    pad_height = target_height - new_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    # 7. Add white padding
    image = ImageOps.expand(
        image,
        border=(left, top, right, bottom),
        fill="white"
    )

    # 8. Convert to NumPy array
    image_array = np.array(image)

    # 9. Normalize pixels
    image_array = image_array.astype(np.float32) / 255.0

    return image_array

def extract_labels(ground_truth):

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    gt_parse = ground_truth.get("gt_parse", {})
    menu = gt_parse.get("menu", [])

    labels = {
        "items": [],
        "quantities": [],
        "prices": []
    }

    # Case 1: menu contains multiple items
    if isinstance(menu, list):

        for item in menu:
            if isinstance(item, dict):
                labels["items"].append(item.get("nm", ""))
                labels["quantities"].append(item.get("cnt", ""))
                labels["prices"].append(item.get("price", ""))

    # Case 2: menu contains one item
    elif isinstance(menu, dict):

        labels["items"].append(menu.get("nm", ""))
        labels["quantities"].append(menu.get("cnt", ""))
        labels["prices"].append(menu.get("price", ""))

    return labels

    for item in gt_parse.get("menu", []):
        labels["items"].append(item.get("nm", ""))
        labels["quantities"].append(item.get("cnt", ""))
        labels["prices"].append(item.get("price", ""))

    return labels