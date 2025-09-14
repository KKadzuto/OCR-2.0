import os

GEMINI_API_KEY = "AIzaSyBuGS2V0jWJOdBsPVY08CeC3oJV-DZFdTo"

DATA_DIR = "хакатон"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
OUTPUT_DIR = "output"
MODELS_DIR = "models"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

OCR_CONFIG = {
    "paddleocr": {
        "use_angle_cls": True,
        "lang": "ru"
    },
    "easyocr": {
        "languages": ["ru", "en"],
        "gpu": False
    }
}

VISION_TRANSFORMER_CONFIG = {
    "donut": {
        "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
        "max_length": 512
    },
    "layoutlmv3": {
        "model_name": "microsoft/layoutlmv3-base",
        "max_length": 512
    }
}

BANK_DOCUMENT_FIELDS = {
    "чек": [
        "дата", "время", "сумма", "номер_чека", "кассир", 
        "магазин", "адрес", "товары", "итого", "налог"
    ],
    "выписка": [
        "дата_выписки", "номер_счета", "валюта", "остаток_на_начало",
        "остаток_на_конец", "операции", "дата_операции", "сумма_операции",
        "назначение_платежа", "контрагент"
    ],
    "договор": [
        "номер_договора", "дата_договора", "стороны", "предмет_договора",
        "сумма", "срок_действия", "условия", "подписи"
    ]
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "extracted_fields": {"type": "object"},
        "confidence_scores": {"type": "object"},
        "raw_text": {"type": "string"},
        "processing_metadata": {
            "type": "object",
            "properties": {
                "ocr_engine": {"type": "string"},
                "vision_transformer": {"type": "string"},
                "llm_model": {"type": "string"},
                "processing_time": {"type": "number"}
            }
        }
    },
    "required": ["document_type", "extracted_fields", "raw_text"]
}
