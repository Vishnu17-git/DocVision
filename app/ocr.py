
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

# Initialize OCR once when the application starts.
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)


def extract_text(image: Image.Image) -> list:
    """Extract text, confidence scores, and pixel-coordinate boxes."""
    image = image.convert("RGB")
    results = ocr.predict(np.array(image))

    extracted = []

    for result in results:
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_boxes"]

        for text, score, box in zip(texts, scores, boxes):
            extracted.append({
                "text": str(text),
                "confidence": float(score),
                "box": [int(value) for value in box],
            })

    return extracted
