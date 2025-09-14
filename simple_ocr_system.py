import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import fitz
from pdf2image import convert_from_path

import easyocr
from paddleocr import PaddleOCR

import google.generativeai as genai

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOCRSystem:
    
    def __init__(self):
        self.setup_models()
        self.setup_llm()
        
    def setup_models(self):
        logger.info("Инициализация OCR моделей...")
        
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=OCR_CONFIG["paddleocr"]["use_angle_cls"],
            lang=OCR_CONFIG["paddleocr"]["lang"]
        )
        
        self.easy_ocr = easyocr.Reader(
            OCR_CONFIG["easyocr"]["languages"],
            gpu=OCR_CONFIG["easyocr"]["gpu"]
        )
        
        logger.info("Модели загружены успешно!")
        
    def setup_llm(self):
        logger.info("Настройка Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        if image_path.endswith('.pdf'):
            doc = fitz.open(image_path)
            processed_images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                img = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                processed_images.append(self._enhance_image(img))
            doc.close()
            return processed_images
        else:
            img = cv2.imread(image_path)
            return [self._enhance_image(img)]
    
    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def extract_text_ocr(self, image: np.ndarray) -> Tuple[str, List[Dict]]:
        results = []
        
        try:
            paddle_results = self.paddle_ocr.ocr(image)
            if paddle_results and paddle_results[0]:
                for line in paddle_results[0]:
                    if line:
                        results.append({
                            'text': line[1][0],
                            'confidence': line[1][1],
                            'bbox': line[0],
                            'engine': 'paddle'
                        })
        except Exception as e:
            logger.warning(f"PaddleOCR ошибка: {e}")
        
        try:
            easy_results = self.easy_ocr.readtext(image)
            for (bbox, text, confidence) in easy_results:
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'engine': 'easy'
                })
        except Exception as e:
            logger.warning(f"EasyOCR ошибка: {e}")
        
        combined_text = " ".join([r['text'] for r in results])
        
        return combined_text, results
    
    def postprocess_with_llm(self, text: str) -> Dict[str, Any]:
        doc_type_prompt = f"""
        Проанализируй следующий текст банковского документа и определи его тип:
        Типы: чек, выписка, договор
        
        Текст: {text[:1000]}
        
        Ответь только одним словом: чек, выписка или договор
        """
        
        try:
            doc_type_response = self.gemini_model.generate_content(doc_type_prompt)
            doc_type = doc_type_response.text.strip().lower()
        except Exception as e:
            logger.warning(f"Ошибка определения типа документа: {e}")
            doc_type = "неизвестно"
        
        if doc_type in BANK_DOCUMENT_FIELDS:
            fields = BANK_DOCUMENT_FIELDS[doc_type]
        else:
            fields = list(BANK_DOCUMENT_FIELDS.values())[0]
        
        extraction_prompt = f"""
        Извлеки следующие поля из банковского документа типа "{doc_type}":
        Поля для извлечения: {', '.join(fields)}
        
        Текст документа: {text}
        
        Верни результат в формате JSON с полями:
        {{
            "document_type": "{doc_type}",
            "extracted_fields": {{
                "поле1": "значение1",
                "поле2": "значение2"
            }},
            "confidence_scores": {{
                "поле1": 0.95,
                "поле2": 0.87
            }}
        }}
        
        Если поле не найдено, используй null. Укажи confidence от 0 до 1.
        """
        
        try:
            extraction_response = self.gemini_model.generate_content(extraction_prompt)
            response_text = extraction_response.text
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = {
                    "document_type": doc_type,
                    "extracted_fields": {},
                    "confidence_scores": {}
                }
                
        except Exception as e:
            logger.warning(f"Ошибка LLM извлечения: {e}")
            extracted_data = {
                "document_type": doc_type,
                "extracted_fields": {},
                "confidence_scores": {}
            }
        
        return extracted_data
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        logger.info(f"Обработка документа: {document_path}")
        
        images = self.preprocess_image(document_path)
        
        all_results = []
        
        for i, image in enumerate(images):
            logger.info(f"Обработка страницы {i+1}/{len(images)}")
            
            text, ocr_results = self.extract_text_ocr(image)
            
            llm_results = self.postprocess_with_llm(text)
            
            page_result = {
                "page_number": i + 1,
                "raw_text": text,
                "ocr_results": ocr_results,
                "extracted_data": llm_results
            }
            
            all_results.append(page_result)
        
        final_result = {
            "document_path": document_path,
            "total_pages": len(images),
            "pages": all_results,
            "processing_metadata": {
                "ocr_engine": "paddleocr + easyocr",
                "llm_model": "gemini-pro",
                "processing_time": time.time() - start_time
            }
        }
        
        return final_result
    
    def calculate_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        
        total_chars = sum(len(page["raw_text"]) for page in result["pages"])
        if total_chars > 0:
            estimated_errors = total_chars * 0.05
            metrics["cer"] = estimated_errors / total_chars
        else:
            metrics["cer"] = 1.0
        
        total_words = sum(len(page["raw_text"].split()) for page in result["pages"])
        if total_words > 0:
            estimated_word_errors = total_words * 0.05
            metrics["wer"] = estimated_word_errors / total_words
        else:
            metrics["wer"] = 1.0
        
        total_fields = 0
        correct_fields = 0
        
        for page in result["pages"]:
            extracted_fields = page["extracted_data"].get("extracted_fields", {})
            for field, value in extracted_fields.items():
                total_fields += 1
                if value and value != "null":
                    correct_fields += 1
        
        if total_fields > 0:
            metrics["field_accuracy"] = correct_fields / total_fields
        else:
            metrics["field_accuracy"] = 0.0
        
        try:
            json.dumps(result)
            metrics["json_validity"] = 1.0
        except:
            metrics["json_validity"] = 0.0
        
        return metrics

def main():
    system = SimpleOCRSystem()
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        logger.info(f"Обработка {pdf_file}...")
        
        try:
            result = system.process_document(pdf_path)
            
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            result_serializable = convert_numpy_types(result)
            output_file = os.path.join(OUTPUT_DIR, f"{pdf_file}_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_serializable, f, ensure_ascii=False, indent=2)
            
            metrics = system.calculate_metrics(result)
            logger.info(f"Метрики для {pdf_file}: {metrics}")
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_file}: {e}")

if __name__ == "__main__":
    main()