
import re


def to_number(value):
    """Convert a text value into a number when possible."""
    if value is None:
        return None

    cleaned = re.sub(r"[^\d.\-]", "", str(value))

    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def parse_item(text):
 
    text = str(text).strip()

    match = re.match(r"^(\d+)\s+(.+)$", text)

    if match:
        return {
            "name": match.group(2).strip(),
            "quantity": int(match.group(1)),
        }

    return {
        "name": text,
        "quantity": None,
    }


def structure_receipt(entities):
    result = {
        "merchant": None,
        "items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
        "payment_method": None,
        "other_fields": {},
    }

    for entity in entities:
        label = str(entity.get("label", "")).upper()
        text = str(entity.get("text", "")).strip()

        if not text:
            continue

        # Merchant name
        if label in {"STORE", "STORE.NAME", "MERCHANT"}:
            result["merchant"] = text

        # Item names
        elif label in {"MENU.NM", "ITEM", "ITEM.NAME"}:
            result["items"].append(parse_item(text))

        # Item prices
        elif label in {"MENU.PRICE", "ITEM.PRICE"}:
            if result["items"]:
                result["items"][-1]["price"] = to_number(text)

        # Receipt total
        elif label in {"TOTAL", "TOTAL.TOTAL_PRICE"}:
            result["total"] = to_number(text)

        # Subtotal
        elif label in {"SUBTOTAL", "SUB_TOTAL"}:
            result["subtotal"] = to_number(text)

        # Tax
        elif label in {"TAX", "TOTAL.TAX_PRICE"}:
            result["tax"] = to_number(text)

        # Payment method
        elif label in {"CASH", "PAYMENT_METHOD"}:
            result["payment_method"] = text

        else:
            result["other_fields"].setdefault(label, []).append(text)

    # Ensure every item has a price field
    for item in result["items"]:
        item.setdefault("price", None)

    return result
