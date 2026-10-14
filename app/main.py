
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from app.ocr import extract_text
from app.model_service import predict_receipt
from app.structure_output import structure_receipt

app = FastAPI(
    title="DocVision API",
    description="Receipt OCR and structured information extraction",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "DocVision API is running"}


@app.post("/extract")
async def extract_receipt(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG or PNG image",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    try:
        with Image.open(BytesIO(contents)) as source:
            source.verify()

        with Image.open(BytesIO(contents)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image",
        )

    try:
        ocr_results = extract_text(image)
        extraction = predict_receipt(image, ocr_results)
        structured_json = structure_receipt(extraction["entities"])
    except Exception as exc:
        # Log the full traceback during development.
        raise HTTPException(
            status_code=500,
            detail=f"Receipt extraction failed: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "ocr_results": ocr_results,
        "extraction": extraction,
        "structured_json": structured_json,
    }
