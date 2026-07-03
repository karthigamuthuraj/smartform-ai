"""
Production Document Filling System (PDF-only demo build)
Handles PDF form filling with AI-powered field detection
"""

import fitz  # PyMuPDF
from typing import Tuple, List, Dict

from pdf_field_handler import PDFFieldHandler


class ProductionDocumentFillingSystem:
    """
    Universal Document Filling System (PDF-only)

    Handles:
    - Flat PDFs → Detect fields + fill
    - Fillable PDFs → Use existing fields + fill
    - Scanned PDFs → OCR + detect fields + fill
    - Checkboxes and Radio buttons

    Uses PDFFieldHandler for all processing.
    """

    def __init__(
        self,
        doc_intelligence_endpoint: str,
        doc_intelligence_key: str,
        openai_endpoint: str,
        openai_key: str,
        openai_deployment: str,
        openai_api_version: str = "2024-08-01-preview",
        logger = None
    ):
        self.pdf_handler = PDFFieldHandler(
            doc_intelligence_endpoint=doc_intelligence_endpoint,
            doc_intelligence_key=doc_intelligence_key,
            openai_endpoint=openai_endpoint,
            openai_key=openai_key,
            openai_deployment=openai_deployment,
            openai_api_version=openai_api_version
        )

        self.logger = logger

    def process_document(
        self,
        file_bytes: bytes,
        file_type: str,
        user_data,
        debug: bool = True
    ) -> Tuple[bytes, List[Dict], Dict]:
        """
        Main entry point for PDF processing.

        Args:
            file_bytes: Document bytes
            file_type: 'pdf'
            user_data: Dictionary of user data OR natural language string
            debug: Print debug information

        Returns:
            (filled_document_bytes, fields_list, metadata)
        """
        file_type_lower = file_type.lower()

        if file_type_lower != 'pdf':
            raise ValueError(f"Unsupported file type: {file_type}. This demo build supports PDF only.")

        if debug:
            print(f"\n{'='*60}")
            print(f"Document Type Detected: PDF")
            print(f"{'='*60}\n")

        return self._process_pdf(file_bytes, user_data, debug)

    def _process_pdf(
        self,
        pdf_bytes: bytes,
        user_data,
        debug: bool
    ) -> Tuple[bytes, List[Dict], Dict]:
        """Process PDF using PDFFieldHandler"""
        if debug:
            print("Step 1: Analyzing PDF structure...")

        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        existing_fields = self.pdf_handler.extract_existing_fields(pdf_doc)

        if existing_fields:
            if debug:
                print(f"   ✓ Found {len(existing_fields)} existing fillable fields\n")
            fields = existing_fields
        else:
            if debug:
                print("   → No existing fields found")
                print("Step 2: Detecting fields with Azure Document Intelligence...")
            fields = self.pdf_handler.detect_fields_with_azure_di(pdf_bytes, pdf_doc, debug)
            if debug:
                print()

        if debug:
            print(f"Step 3: Mapping user data to {len(fields)} PDF fields...")

        field_mappings = self.pdf_handler.map_data_to_fields(fields, user_data, debug)

        if debug:
            print(f"\nStep 4: Filling PDF document...")

        if existing_fields:
            filled_bytes = self.pdf_handler.fill_existing_pdf(pdf_doc, field_mappings, debug)
        else:
            has_polygons = any(field.get("polygon") for field in fields)

            if has_polygons:
                if debug:
                    print("   Using polygon-based filling with rotation support...")
                filled_bytes = self.pdf_handler.fill_pdf_with_polygons(pdf_bytes, fields, field_mappings, debug)
            else:
                filled_bytes = self.pdf_handler.create_fillable_pdf(pdf_bytes, fields, field_mappings, debug)

        metadata = {
            "total_fields": len(fields),
            "mapped_fields": len(field_mappings),
            "has_existing_fields": bool(existing_fields)
        }

        if debug:
            print(f"\n{'='*60}")
            print(f"PDF Processing Complete!")
            print(f"Total Fields: {metadata['total_fields']}")
            print(f"Mapped Fields: {metadata['mapped_fields']}")
            print(f"{'='*60}\n")

        return filled_bytes, fields, metadata
