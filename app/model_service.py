
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "layoutlmv3-cord-5epoch"

DEVICE = torch.device("cpu")  # Safer default for your 8 GB RAM machine

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

processor = LayoutLMv3Processor.from_pretrained(
    str(MODEL_DIR),
    apply_ocr=False,
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    str(MODEL_DIR)
)
model.to(DEVICE)
model.eval()

ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}


def normalize_boxes(ocr_results, width, height):
    words = []
    boxes = []

    for item in ocr_results:
        word = item["text"].strip()
        if not word:
            continue

        x1, y1, x2, y2 = [int(v) for v in item["box"]]

        x1 = max(0, min(1000, round(x1 * 1000 / width)))
        y1 = max(0, min(1000, round(y1 * 1000 / height)))
        x2 = max(0, min(1000, round(x2 * 1000 / width)))
        y2 = max(0, min(1000, round(y2 * 1000 / height)))

        # Ensure valid coordinate ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        words.append(word)
        boxes.append([x1, y1, x2, y2])

    return words, boxes


def group_bio_entities(words, word_predictions):
    entities = []
    current = None

    def save():
        if current is not None:
            entities.append({
                "label": current["label"],
                "text": " ".join(current["words"]),
                "start_word": current["start_word"],
                "end_word": current["end_word"],
            })

    for word_id, word in enumerate(words):
        label = word_predictions.get(word_id, "O")

        if label == "O":
            save()
            current = None
            continue

        if "-" not in label:
            save()
            current = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            save()
            current = {
                "label": entity_type,
                "words": [word],
                "start_word": word_id,
                "end_word": word_id,
            }
        elif prefix == "I":
            if current is not None and current["label"] == entity_type:
                current["words"].append(word)
                current["end_word"] = word_id
            else:
                # Handle an I-label without a matching B-label
                save()
                current = {
                    "label": entity_type,
                    "words": [word],
                    "start_word": word_id,
                    "end_word": word_id,
                }
        else:
            save()
            current = None

    save()
    return entities



def predict_receipt(image: Image.Image, ocr_results: list):
    image = image.convert("RGB")
    width, height = image.size

    words, boxes = normalize_boxes(ocr_results, width, height)

    if not words:
        return {"entities": [], "fields": {}}

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Capture word alignment before converting the encoding
    word_ids = encoding.word_ids(batch_index=0)

    # Move model inputs to the CPU
    model_inputs = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
    }

    with torch.inference_mode():
        logits = model(**model_inputs).logits

    predicted_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    # Map each word to its first predicted token label
    word_predictions = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_predictions:
            word_predictions[word_id] = ID2LABEL[predicted_ids[token_index]]

    entities = group_bio_entities(words, word_predictions)

    fields = {}
    for entity in entities:
        fields.setdefault(entity["label"], []).append(entity["text"])

    return {
        "entities": entities,
        "fields": fields,
    }


def encoding_word_ids(encoding, words):
    # word_ids() comes from the tokenizer's BatchEncoding.
    # Read it before moving tensors to the device in predict_receipt.
    return encoding.word_ids(batch_index=0)
