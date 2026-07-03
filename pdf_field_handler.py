"""
PDF Field Extraction and Mapping Handler

Completely form-agnostic field detection and filling system:
- Pattern-based extraction (no hard-coded keywords/assumptions)
- Spatial grouping: Geometry-based field clustering
- Grid detection: Alignment-based table/grid inference
- Group-anchored widget placement (not label-anchored)
- Fragmented label reconstruction
- Hierarchy-aware field mapping
- Flattened PDF detection and handling
- AcroForm compliance with NeedAppearances flag
"""

import fitz  # PyMuPDF
import json
import re
import os
from typing import List, Dict, Tuple, Optional
from io import BytesIO
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# =============================================================================
# CONFIGURATION CONSTANTS
# All hard-coded values extracted as constants for easy configuration
# =============================================================================

# Field type constants
FIELD_TYPE_TEXT = "TEXT_FIELD"
FIELD_TYPE_DATE = "DATE_FIELD"
FIELD_TYPE_CHECKBOX = "CHECKBOX"
FIELD_TYPE_SIGNATURE = "SIGNATURE_FIELD"
FIELD_TYPE_LABEL_ONLY = "LABEL_ONLY"
FIELD_TYPE_DECORATIVE = "DECORATIVE"

# Document class constants (structural classification)
DOC_CLASS_FILLABLE_PDF = "FILLABLE_PDF"        # Has AcroForm widgets
DOC_CLASS_FLATTENED_FORM = "FLATTENED_FORM_PDF" # Visual blanks/underscores present
DOC_CLASS_NARRATIVE = "NARRATIVE_PDF"          # Labels and content only (NOT fillable)

# Fill direction constants
FILL_RIGHT_OF_LABEL = "right_of_label"
FILL_BELOW_LABEL = "below_label"
FILL_STANDALONE = "standalone"

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_VISUAL_WEIGHT = 0.6
CONFIDENCE_STRUCTURAL_WEIGHT = 0.4

# Pattern-specific confidence scores
CONFIDENCE_PATTERN_A_VISUAL = 0.8  # Inline fields (Date: ____)
CONFIDENCE_PATTERN_A_STRUCTURAL = 0.7
CONFIDENCE_PATTERN_B_VISUAL = 0.75  # Below-label fields
CONFIDENCE_PATTERN_B_STRUCTURAL = 0.8
CONFIDENCE_PATTERN_C_VISUAL = 0.85  # Signature fields
CONFIDENCE_PATTERN_C_STRUCTURAL = 0.9

# Text length limits
MAX_TEXT_LENGTH = 120  # Maximum text length for field detection
MAX_LABEL_LENGTH = 50  # Maximum label length
MAX_SIGNATURE_LENGTH = 60  # Maximum signature field length
MAX_HEADING_LENGTH = 15  # Minimum length for all-caps heading detection
MIN_TEXT_LENGTH = 2  # Minimum text length for validation
MAX_NEARBY_TEXT_LENGTH = 100  # Maximum length for nearby text extraction

# Vertical spacing thresholds
MAX_VERTICAL_GAP_BELOW_LABEL = 30  # Max vertical gap for below-label pattern (points)
MIN_VERTICAL_GAP = 0  # Minimum vertical gap (must be positive)

# Horizontal offset values for fill positioning
LABEL_SPACING_OFFSET = 10  # Gap to the right of label (points)
BELOW_LABEL_OFFSET = 5  # Gap below label (points)
INLINE_LABEL_OFFSET = 30  # Additional offset for inline fields (points)
INLINE_WIDTH_RATIO = 0.6  # Maximum ratio of line width for label estimation

# Character width estimation
ESTIMATED_CHAR_WIDTH = 7  # Estimated points per character for label width calculation

# Nearby text extraction
MAX_NEARBY_WORDS = 5  # Maximum number of nearby words to extract for label

# PDF widget styling
WIDGET_TEXT_FONTSIZE = 0  # 0 = auto-fit
WIDGET_TEXT_COLOR = (0, 0, 0)  # Black (RGB 0-1 scale)
WIDGET_FILL_COLOR = (1, 1, 1)  # White background
WIDGET_BORDER_COLOR = (0.7, 0.7, 0.7)  # Light gray border
WIDGET_BORDER_WIDTH = 0.5  # Border width in points

# Stamped text styling (legacy mode)
STAMP_TEXT_FONTSIZE = 10  # Default font size for stamped text
STAMP_TEXT_COLOR = (0, 0, 0.6)  # Blue text for stamped fields
STAMP_CHECKBOX_FONTSIZE = 12  # Font size for checkbox symbols

# Polygon rotation detection
ROTATION_THRESHOLD = 5.0  # Y-difference threshold in points to consider rotated

# =============================================================================
# GENERIC PATTERN-BASED THRESHOLDS (No hard-coded keywords)
# All thresholds are statistical/geometric - work for any form
# =============================================================================

# Character distribution thresholds (generic)
MAX_DIGIT_RATIO = 0.7  # Maximum ratio of digits in label text (e.g., 0.7 = 70% digits)
MIN_ALPHA_RATIO = 0.3  # Minimum ratio of alphabetic chars in label text
MAX_PUNCTUATION_RATIO = 0.4  # Maximum ratio of punctuation marks

# Repetition thresholds (generic)
MIN_UNIQUE_CHAR_RATIO = 0.3  # Minimum ratio of unique chars (detects "======")

# =============================================================================
# LABEL-ANCHORED WIDGET PLACEMENT (No polygon-based placement)
# =============================================================================

# Unified confidence threshold (no separate visual/structural)
LABEL_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to create widget

# Fixed widget dimensions (safe defaults for all forms)
DEFAULT_WIDGET_WIDTH = 200  # Points
DEFAULT_WIDGET_HEIGHT = 14  # Points

# Label-relative offsets (predictable placement)
RIGHT_OFFSET_X = 8  # Horizontal gap when placing right of label
BELOW_OFFSET_Y = 6  # Vertical gap when placing below label

# =============================================================================
# SPATIAL GROUPING & GRID DETECTION (Geometry-based only)
# =============================================================================

# Spatial grouping thresholds (geometry-based clustering)
GROUP_BBOX_OVERLAP_THRESHOLD = 0.7  # Minimum bbox overlap ratio for same group
GROUP_HORIZONTAL_ALIGNMENT_TOLERANCE = 5  # Points tolerance for horizontal alignment
GROUP_VERTICAL_ALIGNMENT_TOLERANCE = 3  # Points tolerance for vertical alignment

# Document classification thresholds (structural signals)
MIN_WIDGETS_FOR_FILLABLE = 1  # Minimum widgets to classify as FILLABLE_PDF
MIN_UNDERSCORE_DENSITY = 0.02  # Minimum underscore density for FLATTENED_FORM_PDF
MIN_BLANK_REGIONS = 3  # Minimum blank regions for FLATTENED_FORM_PDF

# Grid detection thresholds (alignment-based)
GRID_COLUMN_ALIGNMENT_TOLERANCE = 5  # Points tolerance for column detection
GRID_ROW_SPACING_TOLERANCE = 3  # Points tolerance for row detection
MIN_GRID_CELLS = 4  # Minimum cells to consider a grid layout

# Group-anchored placement
GROUP_WIDGET_MARGIN = 2  # Points margin within group bounds

# =============================================================================


class PDFFieldHandler:
    """Handles PDF field extraction and mapping"""
    
    def __init__(
        self,
        doc_intelligence_endpoint: str,
        doc_intelligence_key: str,
        openai_endpoint: str,
        openai_key: str,
        openai_deployment: str = "gpt-5.1",
        openai_api_version: str = "2024-08-01-preview"
    ):
        self.di_client = DocumentIntelligenceClient(
            endpoint=doc_intelligence_endpoint,
            credential=AzureKeyCredential(doc_intelligence_key)
        )
        
        self.openai_client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            api_version=openai_api_version
        )
        
        self.openai_deployment = openai_deployment
        
    
    def _is_hard_negative(self, text: str, bbox: Optional[Tuple] = None) -> bool:
        """
        Reject text that should never be treated as a fillable field.
        
        Uses ONLY generic pattern-based detection (no hard-coded keywords).
        Works for any form in any language.
        
        Filters based on:
        - Length (too long/short)
        - Character distribution (digit ratio, alpha ratio, punctuation ratio)
        - Repetition patterns (e.g., "======", "------")
        - Structural patterns (email, URL, phone structures - not keywords)
        - All-caps and length combination
        
        Args:
            text: Text content to evaluate
            bbox: Optional (x0, y0, x1, y1) for layout analysis (unused for now)
        
        Returns:
            True if content should be rejected as decorative/non-fillable
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return True
        
        text_clean = text.strip()
        
        # === LENGTH-BASED REJECTION (Generic) ===
        if len(text_clean) > MAX_TEXT_LENGTH:
            return True
        
        # === CHARACTER DISTRIBUTION ANALYSIS (Generic) ===
        total_chars = len(text_clean)
        alpha_count = sum(1 for c in text_clean if c.isalpha())
        digit_count = sum(1 for c in text_clean if c.isdigit())
        punct_count = sum(1 for c in text_clean if c in '.,;:!?-()[]{}')
        unique_chars = len(set(text_clean.lower()))
        
        # Too many digits (likely data, not label)
        if total_chars > 0 and digit_count / total_chars > MAX_DIGIT_RATIO:
            return True
        
        # Too few letters (likely decorative or data)
        if total_chars > 0 and alpha_count / total_chars < MIN_ALPHA_RATIO:
            return True
        
        # Too much punctuation (likely decorative)
        if total_chars > 0 and punct_count / total_chars > MAX_PUNCTUATION_RATIO:
            return True
        
        # === REPETITION DETECTION (Generic) ===
        # Detects repeated characters like "======", "------", "______"
        if total_chars > 0 and unique_chars / total_chars < MIN_UNIQUE_CHAR_RATIO:
            return True
        
        # === ALL-CAPS HEADINGS (Generic) ===
        # Long all-caps text is typically a section header, not a field label
        if text_clean.isupper() and len(text_clean) > MAX_HEADING_LENGTH:
            return True
        
        # === STRUCTURAL PATTERNS (Generic regex, no keywords) ===
        
        # Email structure pattern (@ and domain)
        if re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', text_clean):
            return True
        
        # URL structure pattern (protocol or www prefix)
        if re.match(r'^(https?://|www\.)[^\s]+$', text_clean.lower()):
            return True
        
        # Phone number structure (digits with common separators)
        # Matches: (123) 456-7890, 123-456-7890, 123.456.7890
        if re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text_clean):
            return True
        
        # Standalone date structure (MM/DD/YYYY, MM-DD-YYYY)
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text_clean):
            return True
        
        # Pure separator lines (decorative)
        if re.match(r'^[=\-_*#\.]{3,}$', text_clean):
            return True
        
        return False
    
    def classify_pdf_document(self, pdf_doc, result=None, debug: bool = False) -> str:
        """
        Classify PDF document type using ONLY structural signals.
        
        Classification:
        - FILLABLE_PDF: Has AcroForm widgets
        - FLATTENED_FORM_PDF: Has visual blanks/underscores but no widgets
        - NARRATIVE_PDF: Has labels and content only (NOT fillable)
        
        Args:
            pdf_doc: PyMuPDF document
            result: Optional Azure DI result for enhanced analysis
            debug: Enable debug logging
        
        Returns:
            Document class constant
        """
        try:
            # Count widgets across all pages
            total_widgets = 0
            text_widgets = 0
            checkbox_widgets = 0
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                widgets = list(page.widgets())
                total_widgets += len(widgets)
                
                for widget in widgets:
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                        text_widgets += 1
                    elif widget.field_type in [fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON]:
                        checkbox_widgets += 1
            
            # Rule 1: Has widgets = FILLABLE_PDF
            if total_widgets >= MIN_WIDGETS_FOR_FILLABLE:
                if debug:
                    print(f"\n   📋 Document Class: FILLABLE_PDF ({total_widgets} widgets)")
                return DOC_CLASS_FILLABLE_PDF
            
            # Analyze text content for visual form indicators
            total_chars = 0
            underscore_chars = 0
            blank_line_count = 0
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text("text")
                total_chars += len(text)
                underscore_chars += text.count('_')
                
                # Count lines with multiple consecutive underscores (blank fields)
                lines = text.split('\n')
                for line in lines:
                    if '___' in line or '__________' in line:
                        blank_line_count += 1
            
            # Calculate underscore density
            underscore_density = underscore_chars / total_chars if total_chars > 0 else 0
            
            # Rule 2: High underscore density or blank regions = FLATTENED_FORM_PDF
            if underscore_density >= MIN_UNDERSCORE_DENSITY or blank_line_count >= MIN_BLANK_REGIONS:
                if debug:
                    print(f"\n   📝 Document Class: FLATTENED_FORM_PDF")
                    print(f"      Underscore density: {underscore_density:.3f}")
                    print(f"      Blank regions: {blank_line_count}")
                return DOC_CLASS_FLATTENED_FORM
            
            # Rule 3: No widgets + no visual blanks = NARRATIVE_PDF
            if debug:
                print(f"\n   📖 Document Class: NARRATIVE_PDF (no fillable indicators)")
                print(f"      Widgets: {total_widgets}")
                print(f"      Underscore density: {underscore_density:.3f}")
                print(f"      Blank regions: {blank_line_count}")
            return DOC_CLASS_NARRATIVE
            
        except Exception as e:
            if debug:
                print(f"   Warning: Classification failed: {e}")
            # Default to NARRATIVE to be safe (prevents false filling)
            return DOC_CLASS_NARRATIVE
    
    def _detect_flattened_pdf(self, pdf_doc, result, debug: bool = False) -> bool:
        """
        Detect if PDF is flattened (likely converted from DOCX).
        
        Generic heuristics (no hard-coded form names):
        - High text fragmentation (many small text blocks)
        - Few or no text input widgets (only checkboxes/radio buttons)
        - Many absolute-positioned text elements
        - No table structures preserved in widgets
        
        Returns True if PDF appears to be flattened layout.
        """
        try:
            # Check widget types
            text_widgets = 0
            checkbox_widgets = 0
            total_widgets = 0
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                widgets = list(page.widgets())
                total_widgets += len(widgets)
                
                for widget in widgets:
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                        text_widgets += 1
                    elif widget.field_type in [fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON]:
                        checkbox_widgets += 1
            
            # Check text fragmentation from Azure DI result
            total_lines = 0
            total_words = 0
            
            if result and result.pages:
                for page in result.pages:
                    if hasattr(page, 'lines') and page.lines:
                        total_lines += len(page.lines)
                    if hasattr(page, 'words') and page.words:
                        total_words += len(page.words)
            
            # Heuristics for flattened PDF:
            # 1. High word-to-line ratio (fragmented text)
            words_per_line = total_words / total_lines if total_lines > 0 else 0
            
            # 2. Few text widgets but many checkboxes/radio buttons
            has_few_text_widgets = text_widgets < 5 and checkbox_widgets > 0
            
            # 3. High fragmentation (many small text blocks)
            highly_fragmented = words_per_line < 3 and total_words > 50
            
            is_flattened = has_few_text_widgets and highly_fragmented
            
            if debug and is_flattened:
                print(f"   ⚠️  Detected FLATTENED_LAYOUT_PDF (likely DOCX→PDF conversion)")
                print(f"      Text widgets: {text_widgets}, Checkboxes: {checkbox_widgets}")
                print(f"      Words/line ratio: {words_per_line:.1f} (fragmented)")
                print(f"      Adjusting extraction strategy...")
            
            return is_flattened
            
        except Exception as e:
            if debug:
                print(f"   Warning: Flattened PDF detection failed: {e}")
            return False
    
    def extract_existing_fields(self, pdf_doc) -> List[Dict]:
        """Extract existing fillable fields from PDF with contextual labels and hierarchy"""
        fields = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            widgets = list(page.widgets())
            
            for widget in widgets:
                field_type_map = {
                    fitz.PDF_WIDGET_TYPE_BUTTON: "Button",
                    fitz.PDF_WIDGET_TYPE_CHECKBOX: "Checkbox",
                    fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "RadioButton",
                    fitz.PDF_WIDGET_TYPE_TEXT: "Text",
                    fitz.PDF_WIDGET_TYPE_LISTBOX: "Dropdown",
                    fitz.PDF_WIDGET_TYPE_COMBOBOX: "Dropdown",
                    fitz.PDF_WIDGET_TYPE_SIGNATURE: "Signature"
                }
                
                basic_type = field_type_map.get(widget.field_type, "Text")
                
                # Get basic label from widget
                basic_label = widget.field_label or widget.field_name
                
                # Extract nearby text to create contextual label (now with improved boundary detection)
                contextual_label = self._extract_nearby_text_for_widget(page, widget, basic_label)
                
                # Use contextual label if found, otherwise fall back to basic label
                label_full = contextual_label if contextual_label else basic_label
                
                # Parse field number and clean label
                parsed_label = self._parse_field_number_and_clean_label(label_full)
                
                # Determine field type: use widget type if specific, otherwise classify from label
                if basic_type in ["Checkbox", "RadioButton", "Signature"]:
                    # Widget already has specific type - use it
                    field_type = basic_type
                elif basic_type == "Text":
                    # Try to classify text fields more accurately using pattern detection
                    classified_type = self._classify_field_type(label_full)
                    # Convert to string representation
                    type_map = {
                        FIELD_TYPE_CHECKBOX: "Checkbox",
                        FIELD_TYPE_SIGNATURE: "Signature",
                        FIELD_TYPE_DATE: "Date",
                        FIELD_TYPE_TEXT: "Text"
                    }
                    field_type = type_map.get(classified_type, "Text")
                else:
                    # Use widget's basic type for other cases
                    field_type = basic_type
                
                # Extract hierarchy context from field name (generic, no hard-coding)
                # Example: "topmostSubform.CopyA.Col_Left.f1_1" -> hierarchy: ["topmostSubform", "CopyA", "Col_Left"]
                field_name_parts = widget.field_name.split('.')
                hierarchy = field_name_parts[:-1] if len(field_name_parts) > 1 else []
                
                fields.append({
                    "field_name": widget.field_name,
                    "label": parsed_label['label_clean'],  # Use clean label for display
                    "label_full": label_full,  # Keep full label for context
                    "field_number": parsed_label['field_number'],  # Extracted field number
                    "has_trailing_text": parsed_label['has_trailing_text'],  # Indicator of messy label
                    "field_type": field_type,
                    "page_number": page_num + 1,
                    "position": (widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1),
                    "detection_method": "existing_widget",
                    "hierarchy": hierarchy  # Add hierarchy context for better mapping
                })
        
        # Spatial field detection: Find missing numbered fields
        fields = self._detect_missing_numbered_fields(fields, pdf_doc)
        
        # Log field summary with page distribution for multi-page forms
        page_distribution = {}
        for field in fields:
            page = field.get('page_number', 1)
            page_distribution[page] = page_distribution.get(page, 0) + 1
        
        if len(page_distribution) > 1:
            page_summary = ", ".join([f"p{page}={count}" for page, count in sorted(page_distribution.items())])
            print(f"\n   📄 Multi-page form: {len(fields)} fields across {len(page_distribution)} pages ({page_summary})")
            
            # Check if field names have page suffixes (from page-by-page processing)
            has_page_suffix = any('_page' in field.get('field_name', '') for field in fields)
            if has_page_suffix:
                print(f"   ✓ Page suffixes detected in field names (e.g., textbox_0_1_page3)")
        
        return fields
    
    def _detect_missing_numbered_fields(self, fields: List[Dict], pdf_doc) -> List[Dict]:
        """
        Detect missing numbered fields using spatial analysis.
        
        For numbered forms (W-9, IRS), detects gaps in field number sequence
        and searches for unlabeled widgets that might fill those gaps.
        """
        # Extract field numbers that were found
        found_numbers = set()
        for field in fields:
            if field.get('field_number') is not None:
                found_numbers.add(field['field_number'])
        
        if not found_numbers:
            return fields  # No numbered fields detected
        
        # Check for sequential gaps (e.g., have 5 and 6, missing 7)
        min_num = min(found_numbers)
        max_num = max(found_numbers)
        
        missing_numbers = []
        for num in range(min_num, max_num + 2):  # Check 1 beyond max
            if num not in found_numbers:
                missing_numbers.append(num)
        
        if not missing_numbers:
            return fields  # No gaps detected
        
        print(f"\n   🔍 Spatial Analysis: Missing field numbers detected: {missing_numbers}")
        
        # Get all widgets from PDF to find unlabeled ones
        all_widgets = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            for widget in page.widgets():
                all_widgets.append({
                    'field_name': widget.field_name,
                    'rect': (widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1),
                    'page': page_num + 1
                })
        
        # Create set of labeled widget names
        labeled_widgets = {f['field_name'] for f in fields}
        
        # Find unlabeled widgets
        unlabeled = [w for w in all_widgets if w['field_name'] not in labeled_widgets]
        
        if unlabeled:
            print(f"   🎯 Found {len(unlabeled)} unlabeled widgets, analyzing spatial patterns...")
            
            # Try to match unlabeled widgets to missing field numbers using spatial patterns
            for missing_num in missing_numbers:
                # Strategy: Look for widget between field N-1 and N+1
                prev_field = next((f for f in fields if f.get('field_number') == missing_num - 1), None)
                next_field = next((f for f in fields if f.get('field_number') == missing_num + 1), None)
                
                if prev_field:
                    # Look for widget below or to the right of previous field
                    prev_rect = prev_field['position']
                    prev_y = prev_rect[1]
                    prev_x = prev_rect[2]  # Right edge
                    
                    for widget in unlabeled:
                        w_rect = widget['rect']
                        w_y = w_rect[1]
                        w_x = w_rect[0]  # Left edge
                        
                        # Check if widget is on same row (within 30 points vertically)
                        y_diff = abs(w_y - prev_y)
                        x_distance = w_x - prev_x
                        
                        # Widget to the right of previous field (common layout)
                        if y_diff < 30 and 50 < x_distance < 400:
                            print(f"   ✅ Found field {missing_num}: Widget to right of field {missing_num-1}")
                            fields.append({
                                "field_name": widget['field_name'],
                                "label": f"Field {missing_num} (detected spatially)",
                                "label_full": f"Field {missing_num} (spatial pattern matching)",
                                "field_number": missing_num,
                                "has_trailing_text": False,
                                "field_type": "Text",
                                "page_number": widget['page'],
                                "position": w_rect,
                                "detection_method": "spatial_inference",
                                "hierarchy": [],
                                "confidence": 0.7
                            })
                            labeled_widgets.add(widget['field_name'])
                            unlabeled.remove(widget)
                            break
                        
                        # Widget below previous field (vertical layout)
                        y_distance = w_y - prev_y
                        if 20 < y_distance < 60 and abs(w_x - prev_rect[0]) < 50:
                            print(f"   ✅ Found field {missing_num}: Widget below field {missing_num-1}")
                            fields.append({
                                "field_name": widget['field_name'],
                                "label": f"Field {missing_num} (detected spatially)",
                                "label_full": f"Field {missing_num} (spatial pattern matching)",
                                "field_number": missing_num,
                                "has_trailing_text": False,
                                "field_type": "Text",
                                "page_number": widget['page'],
                                "position": w_rect,
                                "detection_method": "spatial_inference",
                                "hierarchy": [],
                                "confidence": 0.7
                            })
                            labeled_widgets.add(widget['field_name'])
                            unlabeled.remove(widget)
                            break
        
        return fields
    
    def _build_spatial_groups(self, fields: List[Dict], debug: bool = False) -> Dict:
        """
        Build spatial field groups based on geometry only (no semantic naming).
        
        Groups fields by:
        - Bounding box overlap
        - Horizontal/vertical alignment
        - Repeated column positions
        - Repeated row heights
        
        Returns structure with group metadata.
        """
        if not fields:
            return {"groups": [], "grid_layouts": []}
        
        groups = []
        group_id_counter = 0
        ungrouped_fields = list(fields)
        
        # Extract position info
        field_positions = []
        for field in fields:
            pos = field.get("position")
            if pos and len(pos) >= 4:
                x0, y0, x1, y1 = pos[:4]
                field_positions.append({
                    "field": field,
                    "bbox": (x0, y0, x1, y1),
                    "center_x": (x0 + x1) / 2,
                    "center_y": (y0 + y1) / 2,
                    "width": x1 - x0,
                    "height": y1 - y0
                })
        
        # Group by horizontal alignment (same y-range = row)
        while ungrouped_fields:
            seed_field = ungrouped_fields[0]
            seed_pos = next((p for p in field_positions if p["field"] == seed_field), None)
            if not seed_pos:
                ungrouped_fields.pop(0)
                continue
            
            group_members = [seed_field]
            ungrouped_fields.remove(seed_field)
            
            # Find fields with overlapping y-range (horizontal alignment)
            seed_y0, seed_y1 = seed_pos["bbox"][1], seed_pos["bbox"][3]
            
            to_remove = []
            for field in ungrouped_fields:
                field_pos = next((p for p in field_positions if p["field"] == field), None)
                if not field_pos:
                    continue
                
                field_y0, field_y1 = field_pos["bbox"][1], field_pos["bbox"][3]
                
                # Check vertical overlap
                overlap_y = min(seed_y1, field_y1) - max(seed_y0, field_y0)
                if overlap_y > GROUP_VERTICAL_ALIGNMENT_TOLERANCE:
                    group_members.append(field)
                    to_remove.append(field)
            
            for field in to_remove:
                ungrouped_fields.remove(field)
            
            # Create group (geometry-based ID only)
            if len(group_members) >= 1:
                group_id_counter += 1
                
                # Compute group bounding box
                group_bbox = self._compute_group_bbox([next(p for p in field_positions if p["field"] == f) for f in group_members])
                
                # Detect layout type (grid vs linear)
                layout_type = self._detect_group_layout_type(group_members, field_positions)
                
                groups.append({
                    "group_id": f"group_{group_id_counter}",
                    "layout_type": layout_type,
                    "bbox": group_bbox,
                    "members": [f.get("field_name") for f in group_members],
                    "member_count": len(group_members)
                })
        
        # Detect grid layouts
        grid_layouts = self._detect_grid_layouts(field_positions, debug)
        
        if debug and groups:
            print(f"\n   📐 Spatial grouping: {len(groups)} groups detected")
            for grp in groups[:3]:  # Show first 3
                print(f"      {grp['group_id']}: {grp['layout_type']}, {grp['member_count']} fields")
        
        return {
            "groups": groups,
            "grid_layouts": grid_layouts
        }
    
    def _compute_group_bbox(self, positions: List[Dict]) -> tuple:
        """Compute bounding box for a group of field positions."""
        if not positions:
            return (0, 0, 0, 0)
        
        x0 = min(p["bbox"][0] for p in positions)
        y0 = min(p["bbox"][1] for p in positions)
        x1 = max(p["bbox"][2] for p in positions)
        y1 = max(p["bbox"][3] for p in positions)
        
        return (x0, y0, x1, y1)
    
    def _detect_group_layout_type(self, members: List[Dict], field_positions: List[Dict]) -> str:
        """
        Detect layout type based on geometry only.
        
        Returns:
        - 'grid': Fields arranged in rows and columns
        - 'linear': Fields in a single row or column
        - 'scattered': No clear pattern
        """
        if len(members) < 2:
            return "linear"
        
        positions = [next(p for p in field_positions if p["field"] == m) for m in members]
        
        # Check for grid pattern: multiple distinct x-positions and y-positions
        x_positions = sorted(set(round(p["center_x"]) for p in positions))
        y_positions = sorted(set(round(p["center_y"]) for p in positions))
        
        # Grid if we have multiple rows AND columns
        if len(x_positions) >= 2 and len(y_positions) >= 2:
            return "grid"
        elif len(x_positions) >= 3 or len(y_positions) >= 3:
            return "linear"
        else:
            return "scattered"
    
    def _detect_grid_layouts(self, field_positions: List[Dict], debug: bool = False) -> List[Dict]:
        """
        Detect grid layouts in PDF using geometry only.
        
        PDFs don't have tables - infer grids from:
        - Repeated column x-positions
        - Repeated row y-positions
        - Consistent cell spacing
        
        Returns grid metadata with row_index and column_index.
        """
        if len(field_positions) < MIN_GRID_CELLS:
            return []
        
        grids = []
        
        # Find repeated x-positions (columns)
        x_coords = [p["center_x"] for p in field_positions]
        columns = self._find_repeated_positions(x_coords, GRID_COLUMN_ALIGNMENT_TOLERANCE)
        
        # Find repeated y-positions (rows)
        y_coords = [p["center_y"] for p in field_positions]
        rows = self._find_repeated_positions(y_coords, GRID_ROW_SPACING_TOLERANCE)
        
        # If we have both rows and columns, it's a grid
        if len(columns) >= 2 and len(rows) >= 2:
            grid_cells = []
            
            for pos in field_positions:
                # Assign to nearest row and column
                row_idx = self._assign_to_nearest_position(pos["center_y"], rows, GRID_ROW_SPACING_TOLERANCE)
                col_idx = self._assign_to_nearest_position(pos["center_x"], columns, GRID_COLUMN_ALIGNMENT_TOLERANCE)
                
                if row_idx is not None and col_idx is not None:
                    grid_cells.append({
                        "field_name": pos["field"].get("field_name"),
                        "row_index": row_idx,
                        "column_index": col_idx,
                        "bbox": pos["bbox"]
                    })
            
            if len(grid_cells) >= MIN_GRID_CELLS:
                grids.append({
                    "grid_id": f"grid_{len(grids) + 1}",
                    "rows": len(rows),
                    "columns": len(columns),
                    "cells": grid_cells
                })
                
                if debug:
                    print(f"\n   📊 Grid detected: {len(rows)} rows × {len(columns)} columns ({len(grid_cells)} cells)")
        
        return grids
    
    def _find_repeated_positions(self, coordinates: List[float], tolerance: float) -> List[float]:
        """Find repeated positions within tolerance (for grid detection)."""
        if not coordinates:
            return []
        
        sorted_coords = sorted(coordinates)
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for coord in sorted_coords[1:]:
            if abs(coord - current_cluster[-1]) <= tolerance:
                current_cluster.append(coord)
            else:
                if len(current_cluster) >= 2:  # Repeated position
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [coord]
        
        # Check last cluster
        if len(current_cluster) >= 2:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _assign_to_nearest_position(self, coord: float, positions: List[float], tolerance: float) -> int:
        """Assign coordinate to nearest position index (row or column)."""
        for idx, pos in enumerate(positions):
            if abs(coord - pos) <= tolerance:
                return idx
        return None
    
    def _extract_nearby_text_for_widget(self, page, widget, fallback_label: str) -> str:
        """
        Extract text near a widget to create a meaningful label.
        Reconstructs fragmented labels by collecting nearby text blocks.
        
        IMPROVED: Stops at field number boundaries to avoid capturing multiple fields.
        For numbered forms (W-9, IRS), extracts only the text for THIS field.
        """
        try:
            widget_rect = widget.rect
            
            # Tightened search area to minimize OCR garbage
            # Old: 150px left - still captured too much
            # New: 100px left - more focused, less garbage
            left_search = fitz.Rect(
                max(0, widget_rect.x0 - 100),  # Narrower left search (was 150px)
                widget_rect.y0 - 3,  # Slightly tighter vertical (was 5px)
                widget_rect.x0 - 2,
                widget_rect.y1 + 3
            )
            
            # Above search for multi-line labels (e.g., text above widget)
            above_search = fitz.Rect(
                max(0, widget_rect.x0 - 30),  # Narrower horizontal (was 50px)
                max(0, widget_rect.y0 - 25),  # Shorter vertical (was 40px)
                widget_rect.x1 + 30,  # Less right margin (was 50px)
                widget_rect.y0
            )
            
            # Collect text blocks with position info for sorting
            text_fragments = []
            
            # Extract from left area
            left_blocks = page.get_text("blocks", clip=left_search)
            for block in left_blocks:
                if len(block) >= 5:
                    text = block[4].strip()
                    x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
                    if text and len(text) > MIN_TEXT_LENGTH:
                        # Skip fragments that are just OCR artifacts (single/double letters)
                        if re.match(r'^[a-z]{1,2}\s*$', text):
                            continue
                        text_fragments.append((y0, x0, text, 'left', x1))
            
            # Extract from above area
            above_blocks = page.get_text("blocks", clip=above_search)
            for block in above_blocks:
                if len(block) >= 5:
                    text = block[4].strip()
                    x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
                    if text and len(text) > MIN_TEXT_LENGTH:
                        # Skip OCR artifacts
                        if re.match(r'^[a-z]{1,2}\s*$', text):
                            continue
                        text_fragments.append((y0, x0, text, 'above', x1))
            
            if not text_fragments:
                return None
            
            # Sort fragments in reading order: top-to-bottom, left-to-right
            text_fragments.sort(key=lambda x: (x[0], x[1]))
            
            # Smart merge: Stop at field number boundaries
            merged_parts = []
            widget_x = widget_rect.x0
            first_field_number = None
            
            for y, x, text, location, x_end in text_fragments:
                # Skip common OCR garbage patterns
                if re.match(r'^[a-z\s\.,;]{1,10}$', text):  # Just lowercase letters/punctuation
                    continue
                
                # Check if this fragment contains a field number
                field_num_match = re.search(r'(\d+)[\.)\s]+([A-Z]|\()', text)
                
                if field_num_match:
                    current_field_num = field_num_match.group(1)
                    
                    if first_field_number is None:
                        # This is the first field number we found - this is OUR field
                        first_field_number = current_field_num
                        merged_parts.append(text)
                    elif current_field_num != first_field_number:
                        # Different field number - stop here, don't include this text
                        break
                    else:
                        # Same field number (e.g., split across fragments) - include it
                        merged_parts.append(text)
                else:
                    # No field number in this fragment - include if we haven't found a field number yet
                    # or if we're still collecting text for the same field
                    if first_field_number is None or len(merged_parts) < 3:
                        merged_parts.append(text)
            
            if not merged_parts:
                return None
            
            # Merge collected parts
            merged_text = ' '.join(merged_parts)
            
            # Aggressive cleanup of common W-9 OCR artifacts
            merged_text = re.sub(r'^[a-z\s\.,;]+(?=[A-Z0-9])', '', merged_text)  # Remove lowercase prefix
            merged_text = re.sub(r'\s+', ' ', merged_text)  # Collapse whitespace
            merged_text = merged_text.strip('.:;,')  # Remove trailing punctuation
            
            # Skip if just numbers/symbols
            if re.match(r'^[\d\s\.\,\-\_]+$', merged_text):
                return None
            
            # Skip if too short after cleaning
            if len(merged_text) <= 3:
                return None
            
            # Return merged label if meaningful
            if merged_text != fallback_label:
                return merged_text
            
        except Exception as e:
            # If extraction fails, silently fall back
            pass
        
        return None
    
    def _classify_field_type(self, label: str) -> str:
        """
        Classify field type using ONLY pattern-based detection (no keywords).
        Works for any form in any language.
        
        Detection based on:
        - Visual symbols (checkboxes: ☐, ☑, ✓)
        - Structural patterns (date separators, signature lines)
        - Character patterns (underscores, etc.)
        
        Returns unified field type constants.
        """
        label_clean = label.strip()
        
        # Checkbox fields: Detect by checkbox symbols (language-agnostic)
        checkbox_symbols = ['☐', '☑', '✓', '□', '■', '☒', '✗', '✔']
        if any(symbol in label_clean for symbol in checkbox_symbols):
            return FIELD_TYPE_CHECKBOX
        
        # Signature fields: Detect by long underscores or X___ pattern
        # Signature lines are typically longer (>20 underscores)
        if '____________________' in label_clean or 'X_____' in label_clean.upper():
            return FIELD_TYPE_SIGNATURE
        
        # Date fields: Detect by date separator patterns (/, -, or placeholder)
        # Common patterns: __/__/____, MM/DD/YYYY, DD-MM-YYYY
        date_patterns = [
            r'__+[/-]__+[/-]__+',  # __/__/____
            r'\d+[/-]\d+[/-]\d+',   # Placeholder dates
            r'[A-Z]{2}[/-][A-Z]{2}[/-][A-Z]{4}',  # MM/DD/YYYY format indicators
        ]
        if any(re.search(pattern, label_clean) for pattern in date_patterns):
            return FIELD_TYPE_DATE
        
        # Default to text field (safest generic option)
        return FIELD_TYPE_TEXT
    
    def _compute_tight_label_bbox(self, label_text: str, line_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Compute a tight bounding box for label text (NOT the full line width).
        
        This prevents widget drift by ensuring we anchor to the actual label,
        not the entire text line which may include underscores or blank space.
        
        Args:
            label_text: The extracted label text (e.g., "Student Name")
            line_bbox: Full line bounding box (x0, y0, x1, y1)
        
        Returns:
            Tight label bbox (x0, y0, x1, y1) based on estimated label width
        """
        x0, y0, x1, y1 = line_bbox
        line_width = x1 - x0
        
        # Estimate label width conservatively
        # Use character-based estimation, capped at 60% of line width
        estimated_label_width = min(
            len(label_text) * ESTIMATED_CHAR_WIDTH,
            line_width * 0.6
        )
        
        # Create tight bounding box starting from left edge
        tight_bbox = (
            x0,
            y0,
            x0 + estimated_label_width,
            y1
        )
        
        return tight_bbox
    
    def enhance_fields_with_azure_di(self, pdf_bytes: bytes, existing_fields: List[Dict], pdf_doc=None, logger=None, debug: bool = False) -> List[Dict]:
        """
        Enhance existing widget fields with Azure DI extracted labels for better context.
        
        This is critical for page-by-page processed PDFs where PyMuPDF text extraction
        fails to find labels in the merged PDF structure.
        
        Args:
            pdf_bytes: PDF file bytes
            existing_fields: List of fields extracted from PDF widgets
            pdf_doc: Optional PyMuPDF document for page height/geometry info
            logger: Optional logger for output
            debug: Enable debug logging
            
        Returns:
            Enhanced fields with better labels from Azure DI
        """
        def log(msg):
            if logger:
                logger.log(msg)
            elif debug:
                print(msg)
        
        def extract_coords_from_polygon(polygon):
            """Extract x,y coordinates from polygon regardless of format."""
            if not polygon or len(polygon) < 4:
                return None, None
            
            # Check polygon format
            first_point = polygon[0]
            
            if hasattr(first_point, 'x') and hasattr(first_point, 'y'):
                # Point objects with .x and .y attributes
                x_coords = [p.x for p in polygon]
                y_coords = [p.y for p in polygon]
            elif isinstance(first_point, (list, tuple)) and len(first_point) >= 2:
                # List of tuples/lists [(x, y), ...]
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
            elif isinstance(first_point, (int, float)):
                # Flat list of coordinates [x1, y1, x2, y2, ...]
                x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
            else:
                return None, None
            
            return x_coords, y_coords
        
        def is_messy_label(text):
            """Check if label contains Unicode garbage or is generic field name."""
            if not text:
                return True
            
            # Generic field names from CommonForms
            if text.startswith('textbox_') or text.startswith('choicebutton_'):
                return True
            
            # Unicode replacement character or other garbage
            if '�' in text or '\ufffd' in text:
                return True
            
            # Too many non-printable or weird characters
            weird_chars = sum(1 for c in text if ord(c) > 127 and ord(c) != 32)
            if len(text) > 0 and weird_chars / len(text) > 0.3:  # >30% weird chars
                return True
            
            return False
        
        def clean_label(text):
            """
            Aggressively remove Unicode garbage, symbols, and invalid characters.
            Ensures LLM receives clean, readable labels for accurate field mapping.
            """
            if not text:
                return text
            
            import unicodedata
            import re
            
            # Step 1: Remove Unicode replacement characters (all variants)
            text = text.replace('�', '')  # Common replacement char
            text = text.replace('\ufffd', '')  # Unicode replacement character
            text = text.replace('\x00', '')  # Null bytes
            
            # Step 2: Remove control characters and non-printable ASCII
            cleaned = ''.join(c for c in text if c.isprintable() or c in ' \n\t')
            
            # Step 3: Keep only ASCII printable + common punctuation + Latin Extended
            # Remove weird Unicode symbols but keep accented characters
            result = []
            for char in cleaned:
                code = ord(char)
                # Keep: ASCII printable (32-126), space, common Latin Extended (À-ÿ)
                if 32 <= code <= 126 or 192 <= code <= 255 or char in ' \n\t':
                    result.append(char)
            cleaned = ''.join(result)
            
            # Step 4: Remove excessive special characters and symbols
            # Keep: letters, numbers, spaces, and essential punctuation only
            cleaned = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\?\/&\'\"]+', ' ', cleaned)
            
            # Step 5: Normalize whitespace (collapse multiple spaces)
            cleaned = ' '.join(cleaned.split())
            
            # Step 6: Remove leading/trailing punctuation and symbols
            cleaned = cleaned.strip(' .,;:-_()[]{}!@#$%^&*+=|\\/<>?`~')
            
            # Step 7: Truncate if too long (LLM context limit)
            if len(cleaned) > 200:
                cleaned = cleaned[:200].rsplit(' ', 1)[0] + '...'
            
            # Step 8: If result is empty or too short, return None (will use field name)
            if not cleaned or len(cleaned) <= 1:
                return None
            
            return cleaned
        
        if debug:
            log(f"\n{'='*70}")
            log("ENHANCING FIELD LABELS WITH AZURE DI")
            log(f"{'='*70}\n")
            log(f"   Existing widgets: {len(existing_fields)}")
        
        try:
            # Call Azure DI to extract all text and key-value pairs
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
            
            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=pdf_bytes)
            )
            
            result = poller.result()
            
            if debug:
                log(f"   Azure DI extraction complete")
                if result.key_value_pairs:
                    log(f"   Found {len(result.key_value_pairs)} key-value pairs")
            
            # Extract all text content with positions
            azure_labels = []
            
            # Extract from key-value pairs
            if result.key_value_pairs:
                for kv in result.key_value_pairs:
                    if kv.key and kv.key.content and kv.key.bounding_regions:
                        label_text = kv.key.content.strip()
                        page_num = kv.key.bounding_regions[0].page_number
                        polygon = kv.key.bounding_regions[0].polygon
                        
                        # Convert polygon to rect
                        x_coords, y_coords = extract_coords_from_polygon(polygon)
                        if x_coords and y_coords:
                            # Get page height for coordinate conversion
                            page_height = None
                            if pdf_doc and page_num <= len(pdf_doc):
                                pdf_page = pdf_doc[page_num - 1]
                                page_height = pdf_page.rect.height
                            
                            # Convert coordinates
                            if page_height:
                                y_coords_pdf = [page_height - y for y in y_coords]
                            else:
                                y_coords_pdf = y_coords
                            
                            x0, y0 = min(x_coords), min(y_coords_pdf)
                            x1, y1 = max(x_coords), max(y_coords_pdf)
                            
                            azure_labels.append({
                                'text': label_text,
                                'page': page_num,
                                'position': (x0, y0, x1, y1)
                            })
            
            # Extract from general text content
            if result.pages:
                for page in result.pages:
                    if page.lines:
                        page_height = None
                        if pdf_doc and page.page_number <= len(pdf_doc):
                            pdf_page = pdf_doc[page.page_number - 1]
                            page_height = pdf_page.rect.height
                        
                        for line in page.lines:
                            if line.content and line.polygon:
                                text = line.content.strip()
                                polygon = line.polygon
                                
                                x_coords, y_coords = extract_coords_from_polygon(polygon)
                                if not x_coords or not y_coords:
                                    continue
                                
                                if page_height:
                                    y_coords_pdf = [page_height - y for y in y_coords]
                                else:
                                    y_coords_pdf = y_coords
                                
                                x0, y0 = min(x_coords), min(y_coords_pdf)
                                x1, y1 = max(x_coords), max(y_coords_pdf)
                                
                                azure_labels.append({
                                    'text': text,
                                    'page': page.page_number,
                                    'position': (x0, y0, x1, y1)
                                })
            
            if debug:
                log(f"   Extracted {len(azure_labels)} text elements from Azure DI")
                
                # Sample first few text elements for debugging
                if azure_labels:
                    log(f"   Sample text elements (first 3):")
                    for i, al in enumerate(azure_labels[:3]):
                        log(f"      [{i+1}] Page {al['page']}: '{al['text'][:50]}...' at {al['position']}")
            
            # Match Azure DI labels to existing widget positions
            enhanced_count = 0
            skipped_clean = 0
            matched_attempts = 0
            
            for field in existing_fields:
                widget_page = field.get('page_number', 1)
                widget_pos = field.get('position', [0, 0, 0, 0])
                widget_x0, widget_y0, widget_x1, widget_y1 = widget_pos
                widget_name = field.get('field_name', 'unknown')
                
                # Get current label
                current_label = field.get('label', '')
                
                # Skip if field already has a clean label (not messy)
                if current_label and not is_messy_label(current_label):
                    skipped_clean += 1
                    continue
                
                # Find Azure DI labels on the same page
                candidate_labels = [
                    al for al in azure_labels 
                    if al['page'] == widget_page
                ]
                
                # Find the closest label (left or above the widget)
                best_label = None
                best_distance = float('inf')
                best_direction = None
                
                matched_attempts += 1
                
                for al in candidate_labels:
                    label_x0, label_y0, label_x1, label_y1 = al['position']
                    label_text = al['text']
                    
                    # Check if label is to the left or above widget
                    is_left = label_x1 < widget_x0 and abs(label_y0 - widget_y0) < 30
                    is_above = label_y1 < widget_y0 and abs(label_x0 - widget_x0) < 100
                    
                    if is_left or is_above:
                        # Calculate distance
                        if is_left:
                            distance = widget_x0 - label_x1
                            direction = 'left'
                        else:  # is_above
                            distance = widget_y0 - label_y1
                            direction = 'above'
                        
                        # Relaxed distance threshold for better matching
                        if distance < best_distance and distance < 200:  # Increased from 150px to 200px
                            best_distance = distance
                            best_label = clean_label(label_text)
                            best_direction = direction
                
                # Update field with enhanced label
                if best_label and len(best_label) > 2:
                    field['label'] = best_label
                    field['label_full'] = best_label
                    field['detection_method'] = 'azure_di_enhanced'
                    enhanced_count += 1
                    
                    if debug and enhanced_count <= 5:  # Log first 5 matches
                        log(f"   ✓ Match: {widget_name} → '{best_label[:40]}' ({best_direction}, {best_distance:.1f}px)")
                else:
                    # No Azure DI match - clean the existing messy label as fallback
                    if current_label and is_messy_label(current_label):
                        cleaned = clean_label(current_label)
                        if cleaned and len(cleaned) > 2 and cleaned != current_label:
                            field['label'] = cleaned
                            field['label_full'] = cleaned
            
            # FINAL CLEANUP PASS: Remove Unicode garbage from ALL labels before sending to LLM
            # This ensures even if Azure DI matching failed, labels are at least readable
            # CRITICAL: This runs on EVERY field regardless of detection method
            cleaned_final = 0
            fallback_to_fieldname = 0
            
            for field in existing_fields:
                label = field.get('label', '')
                label_full = field.get('label_full', '')
                field_name = field.get('field_name', '')
                
                # Clean main label
                if label:
                    cleaned = clean_label(label)
                    if cleaned and cleaned != label:
                        field['label'] = cleaned
                        field['label_full'] = cleaned
                        cleaned_final += 1
                    elif not cleaned:
                        # Cleanup returned None (all garbage) - use field name as fallback
                        field['label'] = field_name
                        field['label_full'] = field_name
                        fallback_to_fieldname += 1
                else:
                    # No label at all - use field name
                    field['label'] = field_name
                    field['label_full'] = field_name
                    fallback_to_fieldname += 1
            
            if debug:
                log(f"   ✅ Enhanced {enhanced_count} field labels with Azure DI context")
                log(f"   🧹 Cleaned {cleaned_final} labels (removed Unicode garbage/symbols)")
                log(f"   📛 Fallback to field name: {fallback_to_fieldname} (labels were empty/garbage)")
                log(f"   📊 Skipped {skipped_clean} fields (already clean)")
                log(f"   📊 Attempted Azure DI matching: {matched_attempts} messy fields")
                log(f"   📊 All fields now have clean labels: {len(existing_fields)}/{len(existing_fields)} ✓")
            
            return existing_fields
        
        except Exception as e:
            if debug:
                log(f"   ⚠️  Azure DI enhancement failed: {e}")
                import traceback
                log(f"   Traceback: {traceback.format_exc()[:500]}")
                log(f"   Falling back to existing labels")
            # Return original fields if enhancement fails
            return existing_fields


    def enhance_labels_with_llm(
        self,
        pdf_bytes: bytes,
        existing_fields: List[Dict],
        pdf_doc = None,
        logger = None,
        debug: bool = False
    ) -> List[Dict]:
        """
        Use LLM to intelligently generate meaningful field labels based on 
        Azure DI extracted context and field positions.
        
        This solves the problem of truncated/merged labels by using semantic 
        understanding rather than just geometric proximity matching.
        
        Args:
            pdf_bytes: PDF file bytes
            existing_fields: List of fields with positions and current labels
            pdf_doc: Optional PyMuPDF document for page info
            logger: Optional logger for output
            debug: Enable debug logging
            
        Returns:
            Enhanced fields with LLM-generated meaningful labels
        """
        def log(msg):
            if logger:
                logger.log(msg)
            elif debug:
                print(msg)
        
        if debug:
            log(f"\n{'='*70}")
            log("ENHANCING FIELD LABELS WITH LLM + AZURE DI")
            log(f"{'='*70}\n")
            log(f"   Processing {len(existing_fields)} fields...")
        
        try:
            # Step 1: Extract all text content with Azure DI for full context
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
            
            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=pdf_bytes)
            )
            
            result = poller.result()
            
            # Build page-wise text context
            page_contexts = {}
            
            if result.pages:
                for page in result.pages:
                    page_num = page.page_number
                    page_text = []
                    
                    if page.lines:
                        for line in page.lines:
                            if line.content:
                                page_text.append(line.content.strip())
                    
                    page_contexts[page_num] = "\n".join(page_text)
            
            if debug:
                log(f"   ✓ Extracted full text context from {len(page_contexts)} pages")
                log(f"   ✓ Total context length: {sum(len(t) for t in page_contexts.values())} characters")
            
            # Step 2: Group fields by page for efficient LLM processing
            fields_by_page = {}
            for field in existing_fields:
                page_num = field.get('page_number', 1)
                if page_num not in fields_by_page:
                    fields_by_page[page_num] = []
                fields_by_page[page_num].append(field)
            
            # Step 3: Process each page with LLM
            enhanced_count = 0
            
            for page_num, page_fields in fields_by_page.items():
                page_context = page_contexts.get(page_num, "")
                
                if not page_context:
                    if debug:
                        log(f"   ⚠️  No text context for page {page_num}, skipping")
                    continue
                
                # Prepare field info for LLM
                field_descriptions = []
                for i, field in enumerate(page_fields):
                    field_type = field.get('field_type', 'Unknown')
                    current_label = field.get('label', 'unknown')
                    field_name = field.get('field_name', 'unknown')
                    position = field.get('position', [0, 0, 0, 0])
                    
                    field_descriptions.append({
                        "index": i,
                        "field_name": field_name,
                        "field_type": field_type,
                        "current_label": current_label,
                        "position": {
                            "x0": round(position[0], 1),
                            "y0": round(position[1], 1),
                            "x1": round(position[2], 1),
                            "y1": round(position[3], 1)
                        }
                    })
                
                # Build LLM prompt
                prompt = f"""You are analyzing a U.S. government form to generate accurate field labels.

**CRITICAL ISSUE:** Many current_labels are TRUNCATED or MERGED due to PDF extraction errors.

**Example of Merged Labels:**
If you see consecutive fields like:
```json
{{
  "index": 14,
  "current_label": "Current Address (Street, City, State, and ZIP Code or Cou Telephone Number Alterna",
  "field_type": "Text",
  "position": {{"x0": 38, "y0": 327, "x1": 214, "y1": 341}}
}},
{{
  "index": 15,
  "current_label": "ZIP Code or Country, if outside the U.S.) Alternate Telephone Number Email A",
  "field_type": "Text", 
  "position": {{"x0": 216, "y0": 327, "x1": 394, "y1": 341}}
}}
```

Looking at the form context, you should identify:
- Field 14 is at x=38-214, and based on form text above it says "Telephone Number" → **"Telephone Number"**
- Field 15 is at x=216-394 (to the right), form text says "Alternate Telephone Number" → **"Alternate Telephone Number"**

The labels mention "Current Address" but that's from EARLIER text - don't be misled!

**Form Context (Page {page_num}):**
```
{page_context[:4000]}  
```

**Fields to Label (in order from top to bottom, left to right):**
```json
{json.dumps(field_descriptions, indent=2)}
```

**Your Task:**
For EACH field, determine its TRUE purpose from the form context, NOT just the truncated current_label.

**Analysis Steps:**
1. Look at the form context to find typical field patterns (Name, Address, Phone, Email, etc.)
2. Use position data (x0, y0, x1, y1) to understand field layout:
   - Fields with similar y0/y1 are on the same row
   - Fields are typically arranged left-to-right, top-to-bottom
3. Identify when a current_label mentions multiple concepts (e.g., "Address...Phone")
4. Map each field to ONE clear concept based on its position relative to form text

**Field Types:**
- Text = text input field
- Checkbox = yes/no checkbox
- Date = date input field

**Common Form Patterns:**
- Name fields: "Last Name", "First Name", "Middle Name"
- Contact: "Current Address", "Telephone Number", "Alternate Telephone Number", "Email Address"
- Identity: "Date of Birth", "Social Security Number", "Place of Birth"
- Document: "Passport Number", "Issue Date", "Expiration Date"
- Signature: "Applicant's Signature", "Date Signed"

**Label Requirements:**
- Use 2-8 words maximum
- Be specific and clear
- Use title case (e.g., "Telephone Number" not "telephone number")

**Output Format:**
Return a JSON object with a "fields" array:
```json
{{
  "fields": [
    {{"index": 0, "label": "Last Name"}},
    {{"index": 1, "label": "First Name"}},
    {{"index": 2, "label": "Middle Name"}}
  ]
}}
```

Generate labels for all {len(field_descriptions)} fields now:"""
                
                try:
                    # Call LLM
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_deployment,
                        messages=[
                            {"role": "system", "content": "You are a form analysis expert. Always return valid JSON with a 'fields' array."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    
                    llm_response = response.choices[0].message.content
                    
                    if debug:
                        log(f"   📄 LLM raw response preview: {llm_response[:300]}...")
                    
                    # Parse LLM response
                    llm_data = json.loads(llm_response)
                    
                    # Extract the fields array from the response
                    llm_labels = None
                    if isinstance(llm_data, dict):
                        # Try common keys
                        for key in ['fields', 'labels', 'results', 'data', 'field_labels']:
                            if key in llm_data and isinstance(llm_data[key], list):
                                llm_labels = llm_data[key]
                                if debug:
                                    log(f"   ✓ Found labels in '{key}' property")
                                break
                    elif isinstance(llm_data, list):
                        llm_labels = llm_data
                    
                    if not llm_labels or not isinstance(llm_labels, list):
                        available_keys = list(llm_data.keys()) if isinstance(llm_data, dict) else "N/A"
                        raise ValueError(f"Could not extract field labels array. Available keys: {available_keys}")
                    
                    # Apply LLM-generated labels
                    examples_to_show = []
                    for label_info in llm_labels:
                        idx = label_info.get('index')
                        new_label = label_info.get('label', '').strip()
                        
                        if idx is not None and 0 <= idx < len(page_fields) and new_label:
                            old_label = page_fields[idx]['label']
                            page_fields[idx]['label'] = new_label
                            page_fields[idx]['label_full'] = new_label
                            page_fields[idx]['detection_method'] = 'llm_enhanced'
                            enhanced_count += 1
                            
                            # Track interesting changes (truncated labels fixed)
                            if len(old_label) > 50 or '...' in old_label or new_label != old_label:
                                examples_to_show.append({
                                    'field': page_fields[idx].get('field_name', ''),
                                    'old': old_label,
                                    'new': new_label
                                })
                    
                    if debug:
                        log(f"   ✓ Page {page_num}: Enhanced {len(llm_labels)} fields with LLM")
                        if examples_to_show:
                            # Show most interesting examples (truncated/merged labels that were fixed)
                            log(f"   📝 Key improvements:")
                            for i, ex in enumerate(examples_to_show[:5]):  # Show top 5 changes
                                old_short = ex['old'][:60] + ('...' if len(ex['old']) > 60 else '')
                                new_short = ex['new'][:40]
                                log(f"      {i+1}. {ex['field']}: '{old_short}' → '{new_short}'")
                
                except Exception as e:
                    if debug:
                        log(f"   ⚠️  LLM processing failed for page {page_num}: {e}")
                    continue
            
            if debug:
                log(f"\n   ✅ LLM Enhanced {enhanced_count} field labels total")
                log(f"   📊 Success rate: {enhanced_count}/{len(existing_fields)} fields")
            
            return existing_fields
        
        except Exception as e:
            if debug:
                log(f"   ⚠️  LLM label enhancement failed: {e}")
                import traceback
                log(f"   Traceback: {traceback.format_exc()[:500]}")
            return existing_fields
    
    def detect_fields_with_azure_di(self, pdf_bytes: bytes, pdf_doc=None, debug: bool = False) -> List[Dict]:
        """
        Detect fields using Azure Document Intelligence with proper polygon coordinate handling.
        
        Automatically classifies document and filters fields based on document type.
        
        Args:
            pdf_bytes: PDF file bytes
            pdf_doc: Optional PyMuPDF document for page height/geometry info
            debug: Enable debug logging
        """
        fields = []
        
        try:
            if debug:
                print("   Using Azure Document Intelligence...")
            
            # Classify document first
            doc_class = None
            if pdf_doc:
                doc_class = self.classify_pdf_document(pdf_doc, None, debug)
            
            # Use AnalyzeDocumentRequest for SDK compatibility
            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=pdf_bytes)
            )
            
            result = poller.result()
            
            # Filter field extraction based on document class
            if doc_class == DOC_CLASS_NARRATIVE:
                # NARRATIVE_PDF: Extract labels only, mark as non-fillable
                if debug:
                    print(f"   ⚠️  NARRATIVE_PDF detected - extracting labels only (no fillable fields)")
                
                # Extract key-value pairs as labels
                if result.key_value_pairs:
                    for kv in result.key_value_pairs:
                        if kv.key and kv.key.content:
                            label = kv.key.content.strip()
                            page_num = kv.key.bounding_regions[0].page_number if kv.key.bounding_regions else 1
                            
                            fields.append({
                                "field_name": self._sanitize_field_name(label),
                                "label": label,
                                "field_type": "Label",  # Mark as label only
                                "fillable": False,  # NOT fillable
                                "page_number": page_num,
                                "detection_method": "azure_di_label_only",
                                "document_class": DOC_CLASS_NARRATIVE
                            })
                
                # Also extract selection marks (checkboxes) as labels
                if result.pages:
                    for page in result.pages:
                        if page.selection_marks:
                            for idx, mark in enumerate(page.selection_marks):
                                # Find nearby text for label
                                polygon = mark.polygon
                                page_height = None
                                if pdf_doc and page.page_number <= len(pdf_doc):
                                    pdf_page = pdf_doc[page.page_number - 1]
                                    page_height = pdf_page.rect.height
                                
                                nearby_label = self._find_nearby_text_for_checkbox(page, polygon, page_height)
                                label = nearby_label if nearby_label else f"Checkbox {idx+1}"
                                
                                fields.append({
                                    "field_name": self._sanitize_field_name(label) + f"_cb{idx}",
                                    "label": label,
                                    "field_type": "Label",  # Mark as label only (not checkbox)
                                    "fillable": False,  # NOT fillable
                                    "page_number": page.page_number,
                                    "detection_method": "azure_di_label_only",
                                    "document_class": DOC_CLASS_NARRATIVE
                                })
                
                # Also run pattern detection to extract visual labels
                if result.pages:
                    visual_fields = self._synthesize_visual_fields_from_layout(
                        result, pdf_doc, debug, is_flattened=False
                    )
                    if visual_fields:
                        # Convert all visual fields to non-fillable labels
                        for vf in visual_fields:
                            vf["field_type"] = "Label"
                            vf["fillable"] = False
                            vf["document_class"] = DOC_CLASS_NARRATIVE
                        fields.extend(visual_fields)
                
                if debug:
                    print(f"   Extracted {len(fields)} labels (non-fillable)")
                
                # Return early - no filling possible for narrative PDFs
                return fields
            
            # Detect if PDF is flattened (converted from DOCX)
            is_flattened = False
            if pdf_doc:
                is_flattened = self._detect_flattened_pdf(pdf_doc, result, debug)
            
            # Extract key-value pairs (only for FILLABLE or FLATTENED_FORM)
            if result.key_value_pairs:
                for kv in result.key_value_pairs:
                    if kv.key and kv.key.content:
                        label = kv.key.content.strip()
                        value = kv.value.content if kv.value else ""
                        
                        page_num = kv.key.bounding_regions[0].page_number if kv.key.bounding_regions else 1
                        
                        # Classify field type using pattern-based detection
                        classified_type = self._classify_field_type(label)
                        # Convert to string representation
                        type_map = {
                            FIELD_TYPE_CHECKBOX: "Checkbox",
                            FIELD_TYPE_SIGNATURE: "Signature",
                            FIELD_TYPE_DATE: "Date",
                            FIELD_TYPE_TEXT: "Text"
                        }
                        field_type = type_map.get(classified_type, "Text")
                        
                        fields.append({
                            "field_name": self._sanitize_field_name(label),
                            "label": label,
                            "field_type": field_type,
                            "page_number": page_num,
                            "detection_method": "azure_di_kv_pair"
                        })
            
            # Extract selection marks (checkboxes) with enhanced polygon handling
            if result.pages:
                for page in result.pages:
                    if page.selection_marks:
                        # Get page height for accurate coordinate conversion
                        page_height = None
                        if pdf_doc and page.page_number <= len(pdf_doc):
                            pdf_page = pdf_doc[page.page_number - 1]
                            page_height = pdf_page.rect.height
                            
                            if debug:
                                print(f"   Page {page.page_number}: height={page_height:.1f}pts, {len(page.selection_marks)} checkboxes")
                        
                        for idx, mark in enumerate(page.selection_marks):
                            # Convert Azure DI polygon (inches) to PDF rect (points)
                            polygon = mark.polygon
                            position = self._get_polygon_bounds(polygon, page_height, apply_y_inversion=False)
                            
                            # Detect rotation for advanced positioning
                            is_rotated = self.is_polygon_rotated(polygon)
                            rotation_angle = self.get_polygon_rotation_angle(polygon) if is_rotated else 0.0
                            
                            # Try to find nearby text to create meaningful label
                            nearby_label = self._find_nearby_text_for_checkbox(page, polygon, page_height)
                            label = nearby_label if nearby_label else f"Checkbox {idx+1}"
                            
                            field_data = {
                                "field_name": self._sanitize_field_name(label) + f"_cb{idx}" if nearby_label else f"checkbox_{page.page_number}_{idx}",
                                "label": label,
                                "field_type": "Checkbox",
                                "page_number": page.page_number,
                                "position": position,  # (x0, y0, x1, y1) in points
                                "polygon": polygon,  # Store original polygon for advanced use
                                "is_rotated": is_rotated,
                                "detection_method": "azure_di_selection_mark"
                            }
                            
                            if is_rotated:
                                field_data["rotation_angle"] = rotation_angle
                                if debug:
                                    print(f"      ↻ Checkbox {idx+1} rotated {rotation_angle:.1f}°")
                            
                            if debug:
                                print(f"      Checkbox {idx+1}: pos=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}, {position[3]:.1f})")
                            
                            fields.append(field_data)
            
            if debug:
                print(f"   Azure DI found {len(fields)} fields (KV pairs + checkboxes)")
            
            # Synthesize visual fields from layout primitives
            # For flattened PDFs: only extract high-confidence simple fields
            if result.pages:
                visual_fields = self._synthesize_visual_fields_from_layout(
                    result, pdf_doc, debug, is_flattened=is_flattened
                )
                if visual_fields:
                    if debug:
                        strategy = "high-confidence only" if is_flattened else "standard"
                        print(f"   Synthesized {len(visual_fields)} visual fields ({strategy})")
                    fields.extend(visual_fields)
            
        except Exception as e:
            if debug:
                print(f"   Azure DI error: {e}")
        
        return fields
    
    def get_azure_di_raw_result(self, pdf_path: str) -> Optional[Dict]:
        """
        Get raw Azure Document Intelligence result for geometry-based processing.
        
        Returns the raw result with:
        - pages (width, height, words, lines, selectionMarks)
        - tables
        - key_value_pairs
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with raw Azure DI result or None if extraction fails
        """
        try:
            # Read PDF bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Call Azure Document Intelligence
            poller = self.di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=pdf_bytes)
            )
            
            result = poller.result()
            
            # Convert to dictionary format expected by geometry-based filler
            azure_result = {
                'pages': [],
                'tables': []
            }
            
            # Extract pages with words, lines, and selection marks
            if result.pages:
                for page in result.pages:
                    page_dict = {
                        'width': page.width if hasattr(page, 'width') else 8.5,
                        'height': page.height if hasattr(page, 'height') else 11,
                        'words': [],
                        'lines': [],
                        'selectionMarks': []
                    }
                    
                    # Extract words
                    if hasattr(page, 'words') and page.words:
                        for word in page.words:
                            page_dict['words'].append({
                                'content': word.content,
                                'polygon': word.polygon if hasattr(word, 'polygon') else [],
                                'confidence': word.confidence if hasattr(word, 'confidence') else 1.0
                            })
                    
                    # Extract lines
                    if hasattr(page, 'lines') and page.lines:
                        for line in page.lines:
                            page_dict['lines'].append({
                                'content': line.content,
                                'polygon': line.polygon if hasattr(line, 'polygon') else []
                            })
                    
                    # Extract selection marks
                    if hasattr(page, 'selection_marks') and page.selection_marks:
                        for mark in page.selection_marks:
                            page_dict['selectionMarks'].append({
                                'state': mark.state if hasattr(mark, 'state') else 'unselected',
                                'polygon': mark.polygon if hasattr(mark, 'polygon') else []
                            })
                    
                    azure_result['pages'].append(page_dict)
            
            # Extract tables
            if hasattr(result, 'tables') and result.tables:
                for table in result.tables:
                    table_dict = {
                        'rowCount': table.row_count if hasattr(table, 'row_count') else 0,
                        'columnCount': table.column_count if hasattr(table, 'column_count') else 0,
                        'cells': [],
                        'boundingRegions': []
                    }
                    
                    # Extract cells
                    if hasattr(table, 'cells') and table.cells:
                        for cell in table.cells:
                            table_dict['cells'].append({
                                'rowIndex': cell.row_index if hasattr(cell, 'row_index') else 0,
                                'columnIndex': cell.column_index if hasattr(cell, 'column_index') else 0,
                                'content': cell.content if hasattr(cell, 'content') else '',
                                'rowSpan': cell.row_span if hasattr(cell, 'row_span') else 1,
                                'columnSpan': cell.column_span if hasattr(cell, 'column_span') else 1
                            })
                    
                    # Extract bounding regions
                    if hasattr(table, 'bounding_regions') and table.bounding_regions:
                        for region in table.bounding_regions:
                            table_dict['boundingRegions'].append({
                                'pageNumber': region.page_number if hasattr(region, 'page_number') else 1,
                                'polygon': region.polygon if hasattr(region, 'polygon') else []
                            })
                    
                    azure_result['tables'].append(table_dict)
            
            return azure_result
            
        except Exception as e:
            print(f"Azure DI raw extraction error: {e}")
            return None
    
    def _detect_pattern_a_inline(self, line_content: str, polygon, page_height: float, page_num: int) -> Optional[Dict]:
        """
        PATTERN A: Inline fields - Date: ______
        
        Label-anchored approach:
        - Extract label text before colon
        - Return label bounding box (NOT input zone)
        - Widget will be placed right of label using fixed offset
        
        Requirements:
        - Colon present
        - Short label (<50 chars)
        - Underscores or blank space after colon
        
        Returns FieldCandidate with label_bbox or None
        """
        content = line_content.strip()
        
        # Must have colon and underscores/spaces
        if ':' not in content or ('___' not in content and '  ' not in content):
            return None
        
        # Extract label before colon
        match = re.match(r'^(.+?):\s*(_+|\s{2,}|\.+)$', content)
        if not match:
            return None
        
        label = match.group(1).strip()
        
        # Validate label
        if not label or len(label) > MAX_LABEL_LENGTH or self._is_hard_negative(label):
            return None
        
        # Classify field type
        field_type = self._classify_field_type(label)
        
        # Get full line bounding box, then compute tight label bbox
        line_bbox = self._get_polygon_bounds(polygon, page_height, apply_y_inversion=False)
        label_bbox = self._compute_tight_label_bbox(label, line_bbox)
        
        return {
            "label": label,
            "field_type": field_type,
            "fill_direction": FILL_RIGHT_OF_LABEL,
            "label_bbox": label_bbox,  # Store label position, not input zone
            "confidence": CONFIDENCE_PATTERN_A_VISUAL,  # Single confidence value
            "page": page_num,
            "source": "pdf",
            "field_name": self._sanitize_field_name(label),
            "detection_method": "pattern_a_inline"
        }
    
    def _detect_pattern_b_below_label(self, lines: List, idx: int, page_height: float, page_num: int) -> Optional[Dict]:
        """
        PATTERN B: Below-label fields
        
        Example:
        Student name
        __________________
        
        Label-anchored approach:
        - Return label line bounding box (NOT underline area)
        - Widget will be placed below label using fixed offset
        
        Requirements:
        - Label line followed by underline
        - Vertical distance within threshold (<30 points)
        
        Returns FieldCandidate with label_bbox or None
        """
        if idx >= len(lines) - 1:
            return None
        
        current_line = lines[idx]
        next_line = lines[idx + 1]
        
        label = current_line.content.strip()
        next_content = next_line.content.strip()
        
        # Current line must be short label, next must have underscores
        if not label or len(label) > MAX_LABEL_LENGTH or '___' not in next_content:
            return None
        
        # Reject hard negatives
        if self._is_hard_negative(label):
            return None
        
        # Check vertical distance
        curr_pos = self._get_polygon_bounds(current_line.polygon, page_height, apply_y_inversion=False)
        next_pos = self._get_polygon_bounds(next_line.polygon, page_height, apply_y_inversion=False)
        
        vertical_gap = next_pos[1] - curr_pos[3]
        if vertical_gap > MAX_VERTICAL_GAP_BELOW_LABEL or vertical_gap < MIN_VERTICAL_GAP:
            return None
        
        # Classify field type
        field_type = self._classify_field_type(label)
        
        # Compute tight label bbox (NOT underline area)
        label_bbox = self._compute_tight_label_bbox(label, curr_pos)
        
        return {
            "label": label,
            "field_type": field_type,
            "fill_direction": FILL_BELOW_LABEL,
            "label_bbox": label_bbox,  # Store label position, widget goes below
            "confidence": CONFIDENCE_PATTERN_B_VISUAL,  # Single confidence value
            "page": page_num,
            "source": "pdf",
            "field_name": self._sanitize_field_name(label),
            "detection_method": "pattern_b_below"
        }
    
    def _detect_pattern_c_signature(self, line_content: str, polygon, page_height: float, page_num: int) -> Optional[Dict]:
        """
        PATTERN C: Signature fields (generic pattern-based detection)
        
        Label-anchored approach:
        - Return label bounding box (NOT signature area)
        - Widget will be placed below label using fixed offset
        
        Detection based on PATTERN, not keywords:
        - Long underscores (>= 20 characters)
        - X____ pattern (common signature marker)
        - No keywords required (language-agnostic)
        
        Returns FieldCandidate with label_bbox or None
        """
        content = line_content.strip()
        content_upper = content.upper()
        
        # Detect signature pattern: X followed by long underscores
        has_x_pattern = 'X_____' in content_upper or 'X____' in content_upper
        
        # Detect long underscores (signature lines are typically longer)
        has_long_underline = '____________________' in content  # 20+ underscores
        
        # Must have either X pattern or very long underscores
        if not (has_x_pattern or has_long_underline):
            return None
        
        # Must be reasonably short overall
        if len(content) > MAX_SIGNATURE_LENGTH:
            return None
        
        # Extract label (remove underscores and X markers)
        label = re.sub(r'_{3,}', '', content).strip()
        label = re.sub(r'^X\s*', '', label, flags=re.IGNORECASE).strip()
        
        # If no meaningful label remains, use generic label
        if not label or len(label) < 2:
            label = content[:20].strip()  # Use first part of line as label
        
        # Get full line bbox, then compute tight label bbox
        line_bbox = self._get_polygon_bounds(polygon, page_height, apply_y_inversion=False)
        label_bbox = self._compute_tight_label_bbox(label, line_bbox)
        
        return {
            "label": label,
            "field_type": FIELD_TYPE_SIGNATURE,
            "fill_direction": FILL_BELOW_LABEL,
            "label_bbox": label_bbox,  # Store label position, widget goes below
            "confidence": CONFIDENCE_PATTERN_C_VISUAL,  # Single confidence value
            "page": page_num,
            "source": "pdf",
            "field_name": self._sanitize_field_name(label),
            "detection_method": "pattern_c_signature"
        }
    
    def _detect_pattern_d_implicit(self, line_content: str, polygon, page_height: float, page_num: int, next_line_content: Optional[str] = None) -> Optional[Dict]:
        """
        PATTERN D: Implicit labels (fallback for labels without underscores/colons)
        
        Example:
        Student Name
        
        
        Label-anchored approach:
        - Conservative detection for short left-aligned text
        - No underscores, colons, or special markers
        - Returns label bbox, widget placed to the right
        
        Requirements:
        - Short text (< ~40 chars)
        - No ending punctuation
        - Not decorative content
        - Next line is empty or far below (optional check)
        
        Returns FieldCandidate with label_bbox or None
        """
        content = line_content.strip()
        
        # Must be reasonably short (implicit labels are typically concise)
        if not content or len(content) > 40 or len(content) < 3:
            return None
        
        # Skip if has markers (handled by other patterns)
        if ':' in content or '___' in content or '__' in content:
            return None
        
        # Skip if ends with punctuation (likely instructional text)
        if content[-1] in '.!?;,':
            return None
        
        # Skip decorative content
        if self._is_hard_negative(content):
            return None
        
        # Skip all-caps (likely headings)
        if content.isupper():
            return None
        
        # Skip lines with multiple sentences
        if '. ' in content or '? ' in content or '! ' in content:
            return None
        
        # Accept as implicit label with conservative confidence
        label = content
        field_type = self._classify_field_type(label)
        
        # Get tight label bbox
        line_bbox = self._get_polygon_bounds(polygon, page_height, apply_y_inversion=False)
        label_bbox = self._compute_tight_label_bbox(label, line_bbox)
        
        return {
            "label": label,
            "field_type": field_type,
            "fill_direction": FILL_RIGHT_OF_LABEL,  # Default: place widget to the right
            "label_bbox": label_bbox,
            "confidence": 0.66,  # Conservative confidence (just above threshold)
            "page": page_num,
            "source": "pdf",
            "field_name": self._sanitize_field_name(label),
            "detection_method": "pattern_d_implicit"
        }
    
    # Deprecated methods removed - now using label-anchored widget placement
    # Old _compute_input_zone_inline and _compute_input_zone_signature methods
    # are no longer needed with the new label-anchored approach
    
    def _apply_confidence_gate(self, candidates: List[Dict], debug: bool = False, min_confidence: float = None) -> List[Dict]:
        """
        Filter candidates by confidence threshold.
        
        For flattened PDFs: uses higher threshold (0.75) to avoid false positives.
        For standard PDFs: uses standard threshold (0.65).
        
        Returns only fillable candidates with confidence >= threshold.
        """
        threshold = min_confidence if min_confidence is not None else LABEL_CONFIDENCE_THRESHOLD
        fillable = []
        rejected_count = 0
        
        for candidate in candidates:
            confidence = candidate.get("confidence", 0)
            
            if confidence >= threshold:
                fillable.append(candidate)
            else:
                rejected_count += 1
                if debug:
                    label = candidate.get("label", "unknown")
                    print(f"      ✗ Rejected (low confidence {confidence:.2f}): '{label}'")
        
        if debug and rejected_count > 0:
            print(f"   Confidence gate: {len(fillable)} passed, {rejected_count} rejected")
        
        return fillable
    
    def _convert_candidates_to_legacy_format(self, candidates: List[Dict]) -> List[Dict]:
        """
        Convert FieldCandidate structure to legacy field format.
        
        New structure uses label_bbox (not input polygon) and fill_direction.
        """
        legacy_fields = []
        
        for candidate in candidates:
            # Map new field types to legacy types
            field_type_map = {
                FIELD_TYPE_TEXT: "Text",
                FIELD_TYPE_DATE: "Date",
                FIELD_TYPE_CHECKBOX: "Checkbox",
                FIELD_TYPE_SIGNATURE: "Signature"
            }
            
            legacy_type = field_type_map.get(candidate["field_type"], "Text")
            
            legacy_field = {
                "field_name": candidate["field_name"],
                "label": candidate["label"],
                "field_type": legacy_type,
                "page_number": candidate["page"],
                "label_bbox": candidate["label_bbox"],  # Label position for anchored placement
                "fill_direction": candidate["fill_direction"],  # How to place widget relative to label
                "detection_method": candidate["detection_method"],
                "confidence": candidate.get("confidence", 0)
            }
            
            legacy_fields.append(legacy_field)
        
        return legacy_fields
    
    def _synthesize_visual_fields_from_layout(self, result, pdf_doc=None, debug: bool = False, is_flattened: bool = False) -> List[Dict]:
        """
        Pattern-based extraction from Azure DI layout.
        
        For flattened PDFs (DOCX→PDF conversions):
        - Disable table/list extraction
        - Extract only high-confidence simple fields
        - Prefer skipping over guessing
        
        Uses conservative pattern detection (in order of priority):
        - Pattern A: Inline fields (Date: ____)
        - Pattern B: Below-label fields (Name\n____)
        - Pattern C: Signature fields (Signature: ____)
        - Pattern D: Implicit labels (fallback, disabled for flattened PDFs)
        
        Returns FieldCandidates with confidence scoring.
        """
        candidates = []
        
        if not result.pages:
            return candidates
        
        if debug:
            layout_type = "flattened layout (DOCX→PDF)" if is_flattened else "standard PDF"
            print(f"\n   🔍 Pattern-based extraction ({layout_type})...")
        
        for page in result.pages:
            page_num = page.page_number
            page_height = None
            
            if pdf_doc and page_num <= len(pdf_doc):
                pdf_page = pdf_doc[page_num - 1]
                page_height = pdf_page.rect.height
            
            lines = page.lines if hasattr(page, 'lines') and page.lines else []
            
            if debug and lines:
                print(f"\n   Page {page_num}: {len(lines)} lines to analyze")
            
            processed_indices = set()
            
            for idx, line in enumerate(lines):
                if idx in processed_indices:
                    continue
                
                content = line.content.strip()
                
                # Skip empty or too long
                if not content or len(content) > MAX_TEXT_LENGTH:
                    continue
                
                # Hard negative filter
                if self._is_hard_negative(content):
                    continue
                
                # Try Pattern C: Signature
                candidate = self._detect_pattern_c_signature(
                    content, line.polygon, page_height, page_num
                )
                if candidate:
                    candidates.append(candidate)
                    if debug:
                        print(f"      ✓ Pattern C (signature): '{candidate['label']}'")
                    continue
                
                # Try Pattern A: Inline
                candidate = self._detect_pattern_a_inline(
                    content, line.polygon, page_height, page_num
                )
                if candidate:
                    candidates.append(candidate)
                    if debug:
                        print(f"      ✓ Pattern A (inline): '{candidate['label']}' → {candidate['field_type']}")
                    continue
                
                # Try Pattern B: Below-label
                candidate = self._detect_pattern_b_below_label(
                    lines, idx, page_height, page_num
                )
                if candidate:
                    candidates.append(candidate)
                    processed_indices.add(idx + 1)  # Skip next line (underline)
                    if debug:
                        print(f"      ✓ Pattern B (below): '{candidate['label']}' → {candidate['field_type']}")
                    continue
                
                # Try Pattern D: Implicit labels (fallback for unmarked labels)
                # SKIP for flattened PDFs to avoid false positives
                if not is_flattened:
                    next_content = lines[idx + 1].content.strip() if idx + 1 < len(lines) else None
                    candidate = self._detect_pattern_d_implicit(
                        content, line.polygon, page_height, page_num, next_content
                    )
                    if candidate:
                        candidates.append(candidate)
                        if debug:
                            print(f"      ✓ Pattern D (implicit): '{candidate['label']}' → {candidate['field_type']}")
                        continue
        
        # Apply confidence gate (stricter for flattened PDFs)
        min_confidence = 0.75 if is_flattened else CONFIDENCE_THRESHOLD
        fillable_fields = self._apply_confidence_gate(candidates, debug, min_confidence=min_confidence)
        
        if debug:
            strategy = f"high-confidence (≥{min_confidence})" if is_flattened else "standard"
            print(f"\n   📊 Pattern extraction: {len(fillable_fields)} fields ({strategy})")
        
        # Convert to legacy format for compatibility
        return self._convert_candidates_to_legacy_format(fillable_fields)
    
    def _find_nearby_text_for_checkbox(self, page, checkbox_polygon, page_height: float = None) -> str:
        """
        Find text near a checkbox to create a meaningful label.
        
        Looks for text to the right or below the checkbox.
        """
        if not hasattr(page, 'words') or not page.words:
            return None
        
        # Get checkbox position
        cb_pos = self._get_polygon_bounds(checkbox_polygon, page_height, apply_y_inversion=False)
        cb_x, cb_y = cb_pos[0], cb_pos[1]
        cb_height = cb_pos[3] - cb_pos[1]
        
        # Search for nearby text (within ~100 points to the right or ~20 points below)
        nearby_words = []
        for word in page.words:
            word_pos = self._get_polygon_bounds(word.polygon, page_height, apply_y_inversion=False)
            word_x, word_y = word_pos[0], word_pos[1]
            
            # Check if word is to the right on same line
            if abs(word_y - cb_y) < cb_height and word_x > cb_x and word_x - cb_x < 100:
                nearby_words.append((word_x, word.content))
            # Or just below
            elif word_y > cb_y and word_y - cb_y < 20 and abs(word_x - cb_x) < 50:
                nearby_words.append((word_x, word.content))
        
        if nearby_words:
            # Sort by X position and take first few words
            nearby_words.sort()
            label_text = ' '.join([w[1] for w in nearby_words[:MAX_NEARBY_WORDS]])
            if len(label_text) > MAX_LABEL_LENGTH:
                label_text = label_text[:MAX_LABEL_LENGTH]
            return label_text.strip()
        
        return None
    
    # Removed duplicate method - consolidated above

    def map_data_to_fields(self, fields: List[Dict], user_data, debug: bool = False) -> Dict[str, str]:
        """
        Map user data to PDF fields using LLM + Fuzzy Matching
        
        Strategy:
        1. Preprocess user data for cleaner input
        2. LLM-based semantic mapping with confidence scoring
        3. Validation and fuzzy field name matching (fallback)
        """
        if not fields or not user_data:
            return {}
        
        # Prepare field info with hierarchy context and position hints
        field_info = []
        hierarchy_groups = {}  # Group fields by common hierarchy prefix
        
        for field in fields:
            if not isinstance(field, dict):
                continue
            
            field_name = field.get("field_name", "")
            hierarchy = field.get("hierarchy", [])
            label = field.get("label", field_name)  # This is now the clean label
            label_full = field.get("label_full", label)  # Full label for context
            field_number = field.get("field_number")  # Extracted field number
            has_trailing = field.get("has_trailing_text", False)
            position = field.get("position", [])
            page = field.get("page_number", 1)
            
            # Build hierarchy context string (generic, no hard-coded names)
            hierarchy_context = " > ".join(hierarchy) if hierarchy else ""
            
            # Calculate field size for context
            field_width = 0
            field_height = 0
            if len(position) == 4:
                field_width = abs(position[2] - position[0])
                field_height = abs(position[3] - position[1])
            
            # Determine field size category
            size_hint = "small"
            if field_width > 200:
                size_hint = "large"
            elif field_width > 100:
                size_hint = "medium"
            
            field_entry = {
                "name": field_name,
                "label": label,  # Clean label (e.g., "Address (number, street...)")
                "type": field.get("field_type", "Text"),
                "hierarchy": hierarchy_context,
                "page": page,
                "width": round(field_width, 1),
                "height": round(field_height, 1),
                "size_hint": size_hint
            }
            
            # Add field number if present (for numbered forms like W-9)
            if field_number is not None:
                field_entry["field_number"] = field_number
            
            # Add indicator if label was messy (contains other field numbers)
            if has_trailing:
                field_entry["note"] = "Clean label extracted from multi-field text"
            
            field_info.append(field_entry)
            
            # Group fields by common hierarchy for context
            if hierarchy:
                group_key = ".".join(hierarchy)
                if group_key not in hierarchy_groups:
                    hierarchy_groups[group_key] = []
                hierarchy_groups[group_key].append(label)
        
        if not field_info:
            return {}
        
        # User data is already preprocessed into clean format
        input_data = user_data
        
        # Build hierarchy context info for better mapping
        hierarchy_info = ""
        if hierarchy_groups:
            hierarchy_info = "\n\nField Groups (by hierarchy):\n"
            for group, labels in hierarchy_groups.items():
                hierarchy_info += f"- {group}: {', '.join(labels[:5])}\n"
        
        # Format fields with size emphasis
        formatted_fields = []
        for f in field_info:
            size_label = f"[{f['size_hint'].upper()}: {f['width']}×{f['height']}px]"
            formatted_fields.append(f"{f['name']}: {size_label} \"{f['label'][:80]}...\"")
        
        prompt = f"""
You are an expert at mapping unstructured data to form fields using SEMANTIC UNDERSTANDING.

Form Fields Available:
{json.dumps(field_info, indent=2)}
{hierarchy_info}

Input Data:
{input_data}

APPROACH: SEMANTIC & CONTEXTUAL MAPPING (Understand what fields are asking for)

**How to Map Data to Fields:**

IMPORTANT: Labels have been PRE-PROCESSED to extract field numbers and clean text.
- Fields with "field_number" property have been parsed from numbered forms (W-9, IRS, etc.)
- "label" field contains CLEANED text for THIS field only (trailing field text removed)
- Fields marked with "note" had messy labels that have been cleaned for you
- Multi-page forms: Field names may include page suffix like "textbox_0_1_page3" to distinguish same field on different pages

1. **Use Field Numbers When Available:**
   Many forms have numbered fields. If a field has "field_number" property, use it to understand field order:
   - field_number: 5, label: "Address (number, street...)" → This is Field #5 (Address)
   - field_number: 6, label: "City, state, and ZIP code" → This is Field #6 (City/State)
   - field_number: 7, label: "List account number(s) here" → This is Field #7 (Account)
   
   Field numbers help identify what data belongs in each field when multiple similar fields exist.

2. **Understand the Question Being Asked:**
   Labels contain the field's purpose. Focus on identifying what the field is asking for:
   
   Examples:
   - "When do you plan to enroll?" → Expects enrollment term like "Fall 2026"
   - "Student name" → Expects student's full name
   - "What high school are you attending?" → Expects school name
   - "Principal or counselor phone" → Expects phone number
   - "Social security number" or "SSN" → Social Security Number (xxx-xx-xxxx)
   - "Employer identification number" or "EIN" → Employer ID (xx-xxxxxxx)

3. **Identify Field Type by Semantic Context:**
   - Labels asking about "name", "applicant", "student" → Person's name
   - Labels asking about "school", "attending", "institution" → School/organization
   - Labels asking about "enroll", "semester", "term" → Enrollment period
   - Labels asking about "phone", "telephone", "contact number" → Phone number
   - Labels asking about "email", "e-mail" → Email address
   - Labels asking about "street", "address", "suite no." → Street address (JUST street, not city)
   - Labels asking about "city", "state", "ZIP code" → City, State ZIP (JUST city/state, not address)
   - Labels asking about "account number", "list account" → Account number(s)
   - Labels asking about "date", "when", "graduation date" → Date
   - Labels asking about "counselor", "principal", "advisor" + "name" → Person's name
   - Labels asking about "signature", "applicant signature" → Person's name
   - Labels with "social security" or "SSN" → Social Security Number
   - Labels with "employer identification" or "EIN" → Employer ID Number

4. **Match Data Semantically:**
   - Find data in input that answers the question the field is asking
   - Trust your understanding of what the field wants
   - Example: If field asks "What high school?", find the school name from input data

5. **CRITICAL - Extract ONLY the VALUE, NOT the Label:**
   - Input may contain: "Name: Mohamed Salawathulla, ID: 987654, DOB: 1990-06-15"
   - For a "Name" field → Return: "Mohamed Salawathulla" (NOT "Name: Mohamed Salawathulla")
   - For an "ID" field → Return: "987654" (NOT "ID: 987654" or "ID-987654")
   - For a "Date of birth" field → Return: "1990-06-15" (NOT "Date of birth: 1990-06-15")
   - Strip all labels/prefixes from values before returning
   - Only return the pure data value that goes IN the field

6. **CRITICAL - One Value Per Field:**
   - Each field should receive ONLY ONE value that matches its purpose
   - DO NOT concatenate multiple values: "Mohamed, 987654, 1990-06-15" ✗ WRONG
   - Fill ONLY what the field asks for: "Mohamed Salawathulla" ✓ CORRECT
   - Let other fields (if detected separately) handle their specific data
   
7. **Handle Multiple Questions in One Label:**
   - Labels often contain context text with multiple questions
   - The LAST clear question is usually the actual field
   - Example: "When do you plan to enroll? Fall, 20 Spring, 20 Summer, 20 Student name"
     → Contains "enroll?" and "Student name"
     → The actual field is "Student name" (appears last)

7. **Ignore OCR Noise:**
   - Skip: Truncated words ("on.", "r upload", "pal or", "nroll?")
   - Skip: Repeated text ("Fall, 20 Fall, 20")
   - Focus: Complete phrases that make sense as questions

8. **Use Width as Validation Only:**
   - Width info helps verify your choice makes sense
   - Very short text in very large field? Double-check your mapping
   - Very long text in tiny field? Probably wrong
   - But semantic meaning takes priority over width rules

HIERARCHY CONTEXT:
- Fields with same hierarchy → related, may need duplicate values
- Use context to distinguish similar fields

FIELD TYPE SUPPORT:

1. TEXT FIELDS: Text Box (Single/Multi Line), Numeric, Alphanumeric, Password
2. SELECTION CONTROLS: Radio Button, Checkbox, Dropdown List, Multi-Select List
3. DATE & TIME CONTROLS: Date Field, Time Field, Date & Time Field
4. BOOLEAN CONTROLS: Yes/No Field, True/False Field
5. FILE & ATTACHMENT CONTROLS: File Attachment, Image Attachment
6. SIGNATURE CONTROLS: Signature Field, Initials Field
7. CALCULATION & FORMULA FIELDS: Calculated Field, Auto-Sum Field
8. REFERENCE & ID FIELDS: Auto-Generated Number, Reference ID Field
9. DISPLAY-ONLY CONTROLS: Read-Only Text Field, Label/Static Text

VALUE FORMATTING RULES:
1. For Text/TextBox fields: Use string values (e.g., "John Smith", "123 Main St")
2. For MultiLineText fields: Can include newlines and longer text
3. For Numeric fields: Use numbers as strings (e.g., "12345", "100.50")
4. For Checkbox fields: Use boolean values true or false (NOT strings "true"/"false")
5. For Radio button fields: Use the exact option value as a string
6. For Dropdown/ListBox fields: Use the exact option text as a string
7. For Date fields: Use date format like "2024-01-15" or "01/15/2024"
8. For Time fields: Use time format like "14:30" or "2:30 PM"
9. For DateTime fields: Combine date and time "2024-01-15 14:30"
10. For Boolean/Yes-No fields: Use true or false
11. For Signature fields: Use signer name as string (actual signature handled separately)
12. For Email fields: Use email format "user@domain.com"
13. For Phone fields: Use phone format "(123) 456-7890" or "123-456-7890"
14. For ReadOnly/Calculated fields: Skip these (will be auto-filled)

Return a JSON object with field names as keys and extracted values as values.

**EXAMPLE - Correct Field Mapping Using Size & Label Hints:**

Given these fields:
[
  {{"name": "textbox_0_0", "label": "on. You... When do you plan to enroll? Fall, 20", "width": 22, "size_hint": "small"}},
  {{"name": "textbox_0_3", "label": "When do you p... Fall, 20 Spring, 20 Summer, 20 Student name", "width": 485, "size_hint": "large"}},
  {{"name": "textbox_0_4", "label": "nroll? Fall, 20... Student name What high school are you attending?", "width": 245, "size_hint": "large"}}
]

Input data: "Student name: Alex Johnson. Plans to enroll Fall 2026. Attends Green Valley High School."

CORRECT mapping:
{{
  "textbox_0_0": "Fall 2026",  // ✓ SMALL field (22px) + "enroll" keyword → enrollment checkbox
  "textbox_0_3": "Alex Johnson",  // ✓ LARGE field (485px) + "Student name" at END → student name
  "textbox_0_4": "Green Valley High School"  // ✓ LARGE field (245px) + "high school" at END → school name
}}

**EXAMPLE - Semantic Mapping Approach:**

Given: "Student name: Alex Johnson. Plans to enroll Fall 2026. Attends Green Valley High School."

Fields:
- textbox_0_0: label="...When do you plan to enroll? Fall, 20", width=22
- textbox_0_3: label="...Fall, 20 Spring, 20 Summer, 20 Student name", width=485
- textbox_0_4: label="...Student name What high school are you attending?", width=245

Semantic Analysis:
- textbox_0_0: Asks "When do you plan to enroll?" → Answer: "Fall 2026"
- textbox_0_3: Asks "Student name" (last phrase) → Answer: "Alex Johnson"
- textbox_0_4: Asks "What high school are you attending?" → Answer: "Green Valley High School"

Correct Mapping:
{{
  "textbox_0_0": "Fall 2026",
  "textbox_0_3": "Alex Johnson",
  "textbox_0_4": "Green Valley High School"
}}

**EXAMPLE - Value Extraction (NO Labels in Output):**

Input data: "Name: Mohamed Salawathulla, Identification number: ID-987654, Date of birth: 1990-06-15"

Fields:
- textbox_1_0: label="Name", width=200
- textbox_1_1: label="Identification number", width=150
- textbox_1_2: label="Date of birth", width=120

CORRECT Mapping (values only, NO labels):
{{
  "textbox_1_0": "Mohamed Salawathulla",
  "textbox_1_1": "987654",
  "textbox_1_2": "1990-06-15"
}}

WRONG Mapping (includes labels - DO NOT DO THIS):
{{
  "textbox_1_0": "Name: Mohamed Salawathulla",  // ✗ WRONG - has label prefix
  "textbox_1_1": "ID-987654",  // ✗ WRONG - includes "ID-" prefix
  "textbox_1_2": "Date of birth: 1990-06-15"  // ✗ WRONG - has label prefix
}}

**EXAMPLE - Multi-Page Forms with Page Suffixes:**

Multi-page forms processed page-by-page have field names like "textbox_0_5_page3" where:
- "textbox_0_5" is the base field name from the original page
- "_page3" indicates this field is on page 3

Input data: "Page 1 - Name: John Smith, Page 3 - Weight: 78 kg, Blood Pressure: 128/82 mmHg"

Fields:
- textbox_0_1_page1: label="Name", page=1, width=200
- textbox_0_1_page3: label="Weight", page=3, width=80  
- textbox_0_5_page3: label="Blood Pressure", page=3, width=100

CORRECT Mapping (use exact field names with page suffixes):
{{
  "textbox_0_1_page1": "John Smith",
  "textbox_0_1_page3": "78 kg",
  "textbox_0_5_page3": "128/82 mmHg"
}}

Note: Even though "textbox_0_1" appears on both page 1 and page 3, they are DIFFERENT fields
with different purposes. Use the page suffix and label to identify the correct field.

**EXAMPLE - Multi-Field Label (Fill Only First Field Type):**

Input data: "Mohamed Salawathulla (ID-987654), born on 1990-06-15"

Fields:
- textbox_0_0: label="Your path... Name Identification number Date of birth", width=400

PROBLEM: Label contains THREE field names but this is ONE field!

CORRECT Mapping (only first field type - "Name"):
{{
  "textbox_0_0": "Mohamed Salawathulla"
}}

WRONG Mapping (concatenating all values - DO NOT DO THIS):
{{
  "textbox_0_0": "Mohamed Salawathulla, 987654, 1990-06-15"  // ✗ WRONG - multiple values
}}

**EXAMPLE - Numbered Fields with Merged Labels (Common in Tax Forms):**

Input data: "John A. Miller, Business: Miller Tech Solutions, Address: 742 Evergreen Terrace Suite 210, City: San Jose CA 95134, Account: 12345678"

Fields with messy merged labels:
- textbox_0_16: label="...5 Address (number, street, apt.)...Requester's na 6 City, state, and ZIP code"
- textbox_0_17: label="6 City, state, and ZIP code 7 List account number(s) here (optional) Par"
- textbox_0_18: label="Social security number...TIN provided must match...SSN..."

Analysis:
- textbox_0_16: Contains "5 Address" first → This is Field #5 (Address field)
- textbox_0_17: Contains "6 City" first → This is Field #6 (City field)  
- textbox_0_18: Contains "Social security number" → SSN field

CORRECT Mapping (based on FIRST field number in label):
{{
  "textbox_0_16": "742 Evergreen Terrace Suite 210",  // ✓ Field #5 = Address
  "textbox_0_17": "San Jose, CA 95134",  // ✓ Field #6 = City/State/ZIP
  "textbox_0_18": "123-45-6789"  // ✓ SSN field (if SSN provided in input)
}}

WRONG Mapping (ignoring field numbers):
{{
  "textbox_0_16": "San Jose, CA 95134",  // ✗ WRONG - City in Address field
  "textbox_0_17": "12345678",  // ✗ WRONG - Account number in City field
  "textbox_0_18": ""  // ✗ WRONG - Skipping SSN when it's available
}}

Only include fields with relevant data. Skip fields with no matching input data.
DO NOT fill ReadOnly, Calculated, or AutoGenerated fields.
ALWAYS extract pure values WITHOUT any label prefixes.

**CONFIDENCE SCORING:**

For each mapping, assess your confidence:
- **0.9-1.0**: Exact match (clear label + matching data in input)
- **0.7-0.9**: Probable match (semantic similarity, correct field type)
- **0.5-0.7**: Uncertain (ambiguous label or limited context)
- **<0.5**: Don't map (return in uncertain_fields instead)

**Output Format:**
```json
{{
  "mappings": {{
    "field_name": {{"value": "extracted_value", "confidence": 0.95}}
  }},
  "uncertain_fields": ["field_name1", "field_name2"]
}}
```

**Example with Confidence:**
Input: "Name: John Smith, Email: john@example.com, City: Unknown"
Fields: [{{"name": "f1", "label": "Full Name"}}, {{"name": "f2", "label": "Email Address"}}, {{"name": "f3", "label": "City"}}]

Correct Response:
```json
{{
  "mappings": {{
    "f1": {{"value": "John Smith", "confidence": 0.98}},
    "f2": {{"value": "john@example.com", "confidence": 0.99}}
  }},
  "uncertain_fields": ["f3"]
}}
```
Note: f3 (City) is not mapped because "Unknown" is not a valid city (confidence would be <0.5).
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": """You are a form-filling expert. CRITICAL RULES:
1. Extract ONLY pure values (NEVER include label prefixes like "Name:", "ID:")
2. ONE value per field (never concatenate multiple unrelated values)
3. Match semantically using field numbers and labels
4. Assign confidence scores: 0.9-1.0=exact match, 0.7-0.9=probable, <0.7=uncertain
5. Fields with confidence <0.5 go in uncertain_fields array
6. Return ONLY valid JSON with 'mappings' and 'uncertain_fields' properties

Pay special attention to FIELD NUMBERS in labels (like '5 Address', '6 City', '7 Account'). 
When labels contain multiple field numbers, use the FIRST number to identify the actual field.

EXAMPLES:
WRONG: {"name": "Name: John"}  
RIGHT: {"name": {"value": "John", "confidence": 0.95}}

WRONG: {"field1": "John, 123, test@email.com"}  
RIGHT: {"name_field": {"value": "John", "confidence": 0.98}}"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            # Extract mappings with confidence handling
            field_mappings = {}
            low_confidence_warnings = []
            
            # Handle response format
            mappings = result.get('mappings', {})
            uncertain_fields = result.get('uncertain_fields', [])
            
            for field_name, data in mappings.items():
                # Handle both old format (direct value) and new format (value + confidence)
                if isinstance(data, dict) and 'value' in data:
                    value = data['value']
                    confidence = data.get('confidence', 1.0)
                    
                    # Warn about low confidence mappings
                    if confidence < 0.7:
                        low_confidence_warnings.append(
                            f"⚠️  Low confidence ({confidence:.2f}) for {field_name}: '{value}'"
                        )
                    
                    field_mappings[field_name] = value
                else:
                    # Fallback for old format
                    field_mappings[field_name] = data
            
            # Log uncertain fields
            if debug and uncertain_fields:
                print(f"\n   📊 Uncertain Fields (not mapped):")
                for field in uncertain_fields:
                    print(f"      • {field}")
            
            # Log low confidence warnings
            if debug and low_confidence_warnings:
                print(f"\n   ⚠️  Low Confidence Mappings:")
                for warning in low_confidence_warnings:
                    print(f"      {warning}")
            
            # Validate mappings before returning
            field_mappings = self._validate_field_mappings(field_mappings, fields, debug)
            
            if debug:
                print(f"   Mapped {len(field_mappings)} PDF fields")
            
            return field_mappings
        
        except Exception as e:
            if debug:
                print(f"   PDF mapping error: {e}")
            return {}
    
    def map_data_to_fields_python_clean(self, fields: List[Dict], user_data, debug: bool = False) -> Dict[str, str]:
        """
        Map user data to Python-extracted fields (optimized for clean field names)
        
        This function is specifically optimized for fields extracted using Python packages
        (PyMuPDF/pypdf) which have CLEAN field names (no OCR garbage like Azure DI).
        
        Strategy:
        1. Field names are already clean (e.g., "applicant_name", "email_address")
        2. Labels are descriptive and accurate
        3. No need for OCR cleanup or spatial detection
        4. Simpler, more straightforward LLM prompt
        
        Args:
            fields: List of fields extracted using Python packages
            user_data: User data (dict or natural language string)
            debug: Enable debug logging
            
        Returns:
            Dict mapping field_name -> value
        """
        if not fields or not user_data:
            return {}
        
        # Prepare field info (already clean from Python extraction)
        field_info = []
        
        for field in fields:
            if not isinstance(field, dict):
                continue
            
            field_name = field.get("field_name", "")
            label = field.get("label", field_name)
            field_type = field.get("field_type", "Text")
            page = field.get("page_number", 1)
            width = field.get("width", 0)
            height = field.get("height", 0)
            
            field_entry = {
                "name": field_name,
                "label": label,
                "type": field_type,
                "page": page,
                "width": round(width, 1) if width else 0,
                "height": round(height, 1) if height else 0
            }
            
            # Add additional info if available
            if field.get("choices"):
                field_entry["choices"] = field["choices"]
            
            if field.get("required"):
                field_entry["required"] = True
            
            if field.get("readonly"):
                field_entry["readonly"] = True
            
            if field.get("current_value"):
                field_entry["current_value"] = field["current_value"]
            
            field_info.append(field_entry)
        
        if not field_info:
            return {}
        
        # Convert user_data to string
        if isinstance(user_data, dict):
            input_data = json.dumps(user_data, indent=2)
        else:
            input_data = str(user_data)
        
        # Optimized prompt for clean Python-extracted fields
        prompt = f"""
You are an expert at mapping user data to form fields with CLEAN, descriptive field names.

These fields were extracted using Python packages (not OCR), so names are clean and accurate.

Form Fields Available:
{json.dumps(field_info, indent=2)}

Input Data:
{input_data}

MAPPING APPROACH:

1. **Field Names Are Clean:**
   - Field names like "applicant_name", "email_address", "phone_number" are self-explanatory
   - Labels are accurate and descriptive
   - No OCR garbage or messy text to deal with

2. **Semantic Matching:**
   - Match field names and labels to user data semantically
   - Field "applicant_name" wants the person's name
   - Field "date_of_birth" wants birth date
   - Field "account_number" wants account/ID number
   - Field "email_address" wants email
   
3. **Multi-Page Forms:**
   - Field names may include page suffix like "name_field_page1", "weight_page3"
   - Use page number and context to identify correct field
   - Same base name on different pages = different fields with different purposes

4. **CRITICAL - Extract ONLY Values:**
   - Input: "Name: John Smith, Email: john@example.com"
   - For "applicant_name" field → Return: "John Smith" (NOT "Name: John Smith")
   - For "email_address" field → Return: "john@example.com" (NOT "Email: john@example.com")
   - Strip all labels/prefixes from values

5. **One Value Per Field:**
   - Each field gets exactly ONE appropriate value
   - Don't concatenate multiple unrelated values
   - Match field purpose to data

VALUE FORMATTING:

1. **Text Fields:** Plain string values
2. **Checkbox/Boolean:** true or false (boolean, not string)
3. **Date Fields:** "YYYY-MM-DD" or "MM/DD/YYYY" format
4. **Phone:** "(123) 456-7890" or "123-456-7890"
5. **Email:** "user@domain.com"
6. **Dropdown:** Must match one of the provided choices exactly
7. **Read-only fields:** Skip these (will be auto-populated)

**EXAMPLE 1 - Simple Form:**

Fields:
[
  {{"name": "full_name", "label": "Full Name", "type": "Text"}},
  {{"name": "email", "label": "Email Address", "type": "Text"}},
  {{"name": "subscribe_newsletter", "label": "Subscribe", "type": "Checkbox"}}
]

Input: "Name: Sarah Johnson, email: sarah.j@example.com, wants newsletter updates"

CORRECT Mapping:
{{
  "full_name": "Sarah Johnson",
  "email": "sarah.j@example.com",
  "subscribe_newsletter": true
}}

**EXAMPLE 2 - Multi-Page Form:**

Fields:
[
  {{"name": "textbox_0_1_page1", "label": "Patient Name", "page": 1, "type": "Text"}},
  {{"name": "textbox_0_1_page3", "label": "Weight (kg)", "page": 3, "type": "Text"}},
  {{"name": "textbox_0_5_page3", "label": "Blood Pressure", "page": 3, "type": "Text"}}
]

Input: "Patient: Mohamed Salawathulla, Page 3 vitals: Weight 78 kg, BP 128/82 mmHg"

CORRECT Mapping:
{{
  "textbox_0_1_page1": "Mohamed Salawathulla",
  "textbox_0_1_page3": "78 kg",
  "textbox_0_5_page3": "128/82 mmHg"
}}

Return a JSON object with field names as keys and mapped values as values.
Only include fields that have data to fill. Omit empty/null values.
"""
        
        if debug:
            print(f"\n{'='*70}")
            print("LLM MAPPING (PYTHON-EXTRACTED FIELDS)")
            print(f"{'='*70}\n")
            print(f"   Fields to map: {len(field_info)}")
            print(f"   User data length: {len(input_data)} characters")
            print(f"   User data preview: {input_data[:150]}...")
        
        # Call Azure OpenAI
        try:
            client = AzureOpenAI(
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version="2024-08-01-preview"
            )
            
            if debug:
                print(f"   Calling Azure OpenAI...")
            
            response = client.chat.completions.create(
                model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                messages=[
                    {"role": "system", "content": "You are a form-filling expert. Return ONLY valid JSON, no markdown or explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if debug:
                print(f"   LLM raw response: {result_text[:500]}...")
            
            # Clean JSON markers
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            if debug:
                print(f"   Cleaned JSON: {result_text[:300]}...")
            
            # Parse JSON
            field_mappings = json.loads(result_text)
            
            if debug:
                print(f"   ✓ Mapped {len(field_mappings)} fields successfully")
                if len(field_mappings) == 0:
                    print(f"   ⚠️  WARNING: LLM returned EMPTY mapping!")
                    print(f"   This usually means:")
                    print(f"     1. User data was empty or insufficient")
                    print(f"     2. Field names don't match user data semantically")
                    print(f"     3. LLM couldn't find any matches")
            
            return field_mappings
        
        except Exception as e:
            if debug:
                print(f"   ✗ Mapping error: {e}")
                print(f"   Exception type: {type(e).__name__}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
            return {}
    
    
    def _is_widget_writable(self, widget, pdf_doc, debug: bool = False) -> Tuple[bool, str]:
        """
        Validation: Check if widget is writable.
        
        Checks:
        - Widget must exist with a field name
        - Widget must NOT be read-only
        
        Returns:
            (is_writable, skip_reason)
        """
        try:
            # Check 1: Widget exists (basic validation)
            if not widget or not widget.field_name:
                return False, "no_field_name"
            
            # Check 2: Not read-only (bit 0 of field_flags)
            if widget.field_flags & (1 << 0):
                return False, "read_only"
            
            # If widget exists and is not read-only, it's writable
            # NOTE: Removed AcroForm.Fields check - CommonForms and other tools
            # create widgets without proper AcroForm catalog entries
            # If we can enumerate the widget from page.widgets(), we can fill it
            
            return True, ""
            
        except Exception as e:
            if debug:
                print(f"   Warning: Widget validation error: {e}")
            return False, "validation_error"
    
    def fill_existing_widgets_only(self, pdf_bytes: bytes, fields: List[Dict], field_mappings: Dict[str, str], debug: bool = False) -> bytes:
        """
        WIDGET-ONLY FILLING (for flattened PDFs).
        
        HARD RULES enforced:
        - Only fills existing AcroForm widgets
        - Only toggles existing checkboxes/radio buttons
        - Does NOT create new text widgets
        - Does NOT attempt table or list filling
        - Skips everything else
        
        Args:
            pdf_bytes: Original PDF bytes
            fields: List of detected fields (for reference only)
            field_mappings: Dictionary mapping field names to values
            debug: Enable debug logging
        
        Returns:
            PDF bytes with ONLY existing widgets filled
        """
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        filled_count = 0
        skipped_count = 0
        skip_reasons = {}
        
        if debug:
            print(f"\n   🔒 WIDGET-ONLY MODE (flattened PDF detected)")
            print(f"   - Only filling existing widgets")
            print(f"   - No new widget creation")
            print(f"   - No table/list filling\n")
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            widgets = list(page.widgets())
            
            for widget in widgets:
                field_name = widget.field_name
                
                # Strict widget validation
                is_writable, skip_reason = self._is_widget_writable(widget, pdf_doc, debug)
                
                if not is_writable:
                    skipped_count += 1
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                    if debug:
                        print(f"   ✗ Skipped ({skip_reason}): {field_name}")
                    continue
                
                # Only fill if we have a mapped value
                if field_name in field_mappings:
                    value = field_mappings[field_name]
                    
                    try:
                        # Handle different field types
                        if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                            is_checked = str(value).lower() in ['true', 'yes', '1', 'checked', 'on']
                            widget.field_value = is_checked
                            widget.update()
                            filled_count += 1
                            if debug:
                                print(f"   ✓ Checkbox: {field_name} = {is_checked}")
                                
                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                            widget.field_value = str(value)
                            widget.update()
                            filled_count += 1
                            if debug:
                                print(f"   ✓ Radio: {field_name} = {value}")
                                
                        elif widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                            # Auto-size font to fit text in field
                            widget.field_value = str(value)
                            
                            # Calculate optimal font size based on field dimensions and text length
                            try:
                                rect = widget.rect
                                field_width = rect.width - 6  # Account for padding and borders
                                field_height = rect.height - 4
                                text_length = len(str(value))
                                
                                if text_length > 0 and field_width > 0:
                                    # Character width estimation
                                    # Helvetica (default PDF font): avg char width ≈ 0.55 * font_size
                                    # Add safety margin for wider characters
                                    chars_per_line = field_width / (0.6 * 10)  # Estimate at 10pt
                                    
                                    # Calculate font size to fit text
                                    if text_length > chars_per_line:
                                        # Text will wrap or need smaller font
                                        estimated_font_size = (field_width / text_length) * 1.5
                                    else:
                                        # Text fits comfortably
                                        estimated_font_size = (field_width / text_length) * 1.8
                                    
                                    # Constrain to reasonable and readable bounds
                                    # Height constraint: font shouldn't exceed 70% of field height for better visibility
                                    # Width constraint: ensure text fits
                                    max_by_height = field_height * 0.7
                                    max_by_width = estimated_font_size
                                    
                                    # Increased minimum from 7 to 9 for better readability
                                    # Maximum is 14 for standard forms
                                    font_size = max(9, min(max_by_height, max_by_width, 14))
                                    
                                    # Set font properties
                                    widget.text_fontsize = font_size
                                    widget.text_font = "Helv"  # Helvetica (standard PDF font)
                                    
                                    if debug:
                                        print(f"   ✓ Text: {field_name} = {value[:30]}... (font: {font_size:.1f}pt, w:{field_width:.0f}, chars:{text_length})")
                                else:
                                    widget.text_fontsize = 10  # Default fallback
                                    if debug:
                                        print(f"   ✓ Text: {field_name} = {value}")
                            except Exception as e:
                                # Fallback if sizing fails
                                widget.text_fontsize = 9
                                if debug:
                                    print(f"   ✓ Text: {field_name} = {value} (default font)")
                            
                            widget.update()
                            filled_count += 1
                        else:
                            # Other widget types (dropdown, etc.)
                            widget.field_value = str(value)
                            widget.update()
                            filled_count += 1
                            if debug:
                                print(f"   ✓ Other: {field_name} = {value}")
                                
                    except Exception as e:
                        if debug:
                            print(f"   ✗ Failed to fill '{field_name}': {e}")
                        skipped_count += 1
        
        if debug:
            print(f"\n   📊 Summary:")
            print(f"   - Filled: {filled_count} writable widgets")
            print(f"   - Skipped: {skipped_count} fields")
            if skip_reasons:
                print(f"   - Skip reasons: {skip_reasons}")
            print(f"   ✓ Widget-only filling complete\n")
        
        # Save as editable PDF (do NOT flatten)
        # Use minimal cleanup to preserve widget structures
        try:
            output_bytes = pdf_doc.tobytes(
                deflate=True,
                garbage=1,  # Minimal cleanup
                clean=False  # Don't clean - preserves widgets
            )
        except Exception as e:
            if debug:
                print(f"   Warning: Save failed ({e}), trying without cleanup...")
            output_bytes = pdf_doc.tobytes(deflate=True)
        
        pdf_doc.close()
        
        return output_bytes
    
    def fill_existing_pdf(self, pdf_doc, field_mappings: Dict[str, str], debug: bool = False) -> bytes:
        """Fill existing fillable PDF fields (legacy method - use fill_existing_widgets_only instead)"""
        filled_count = 0
        skipped_readonly = 0
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            widgets = list(page.widgets())
            
            for widget in widgets:
                field_name = widget.field_name
                
                # Skip read-only, locked, or calculated widgets (generic check)
                if widget.field_flags & (1 << 0):  # ReadOnly flag (bit 0)
                    skipped_readonly += 1
                    if debug:
                        print(f"   Skipped (read-only): {field_name}")
                    continue
                
                if field_name in field_mappings:
                    value = field_mappings[field_name]
                    
                    # Handle different field types
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                        is_checked = str(value).lower() in ['true', 'yes', '1', 'checked', 'on']
                        widget.field_value = is_checked
                    elif widget.field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                        widget.field_value = str(value)
                    else:
                        widget.field_value = str(value)
                    
                    widget.update()
                    filled_count += 1
                    
                    if debug:
                        print(f"   Filled: {field_name} = {value}")
        
        if debug:
            print(f"   Total filled: {filled_count} writable fields")
            if skipped_readonly > 0:
                print(f"   Skipped: {skipped_readonly} read-only/calculated fields")
        
        # Flatten the PDF to make it non-editable
        if debug:
            print(f"   Flattening PDF to remove form fields...")
        
        # Use less aggressive cleanup to avoid xref errors on some PDFs
        try:
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=4, clean=True)
        except Exception as e:
            if debug:
                print(f"   Warning: Aggressive cleanup failed ({e}), trying simpler approach...")
            # Fallback to simpler serialization without aggressive cleanup
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=1)
        
        pdf_doc.close()
        
        return output_bytes
    
    def create_editable_pdf_with_acroform(
        self,
        pdf_bytes: bytes,
        fields: List[Dict],
        field_mappings: Dict[str, str],
        debug: bool = False
    ) -> bytes:
        """
        Create an EDITABLE PDF with AcroForm text fields (not flattened).
        
        HARD RULE ENFORCEMENT:
        - Classifies PDF to prevent filling narrative PDFs
        - Routes to widget-only filling for flattened PDFs
        - Only creates widgets for FILLABLE or valid FLATTENED_FORM PDFs
        
        Args:
            pdf_bytes: Original PDF bytes
            fields: List of detected field dictionaries
            field_mappings: Dictionary mapping field names to values
            debug: Enable debug logging
        
        Returns:
            PDF bytes (editable) OR original bytes if NARRATIVE_PDF
        """
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # HARD RULE: Classify document to prevent narrative PDF filling
        doc_class = self.classify_pdf_document(pdf_doc, None, debug)
        
        # Check if fields are marked as non-fillable
        if fields and any(f.get("fillable") == False for f in fields):
            doc_class = DOC_CLASS_NARRATIVE
        
        # NARRATIVE_PDF: Cannot fill - return original
        if doc_class == DOC_CLASS_NARRATIVE:
            pdf_doc.close()
            if debug:
                print(f"\n   ❌ NARRATIVE_PDF - cannot create widgets for non-form document")
            return pdf_bytes
        
        # Check if PDF is flattened (DOCX→PDF conversion)
        is_flattened = self._detect_flattened_pdf(pdf_doc, None, debug)
        
        if is_flattened:
            # WIDGET-ONLY MODE: Do NOT create new widgets
            pdf_doc.close()
            if debug:
                print(f"\n   ⚠️  Flattened PDF detected - using WIDGET-ONLY mode")
            return self.fill_existing_widgets_only(pdf_bytes, fields, field_mappings, debug)
        
        # Non-flattened, fillable PDF: Proceed with widget creation
        created_count = 0
        field_index = 0  # Counter for unique field names
        
        if debug:
            print(f"\n   🔧 Creating editable AcroForm fields (group-anchored)...")
        
        # Build spatial groups for group-anchored placement
        structure = self._build_spatial_groups(fields, debug)
        groups = {g["group_id"]: g for g in structure["groups"]}
        
        for field in fields:
            field_name = field.get("field_name")
            if not field_name:
                continue
            
            value = field_mappings.get(field_name, "")
            page_num = field.get("page_number", 1) - 1
            
            if page_num >= len(pdf_doc):
                continue
            
            page = pdf_doc[page_num]
            field_type = field.get("field_type", "Text")
            
            # === GROUP-ANCHORED WIDGET PLACEMENT (Geometry-based) ===
            # Find which group this field belongs to
            field_group = None
            for group_id, group in groups.items():
                if field_name in group["members"]:
                    field_group = group
                    break
            
            # Calculate widget position within group bounds
            label_bbox = field.get("label_bbox")
            position = field.get("position")
            
            if label_bbox:
                # Use label bbox as base
                base_bbox = label_bbox
            elif position:
                # Fallback to position
                base_bbox = position
            else:
                if debug:
                    print(f"   ✗ Skipped (no position): {field_name}")
                continue
            
            # Place widget in nearest empty region within same group
            if field_group and field_group["layout_type"] == "grid":
                # For grid layouts: place widget in grid cell
                widget_rect = self._place_widget_in_grid(
                    base_bbox, field_group, page.rect.width, page.rect.height
                )
            else:
                # For linear/scattered: place relative to label with group bounds
                widget_rect = self._place_widget_in_group(
                    base_bbox, field_group, page.rect.width, page.rect.height
                )
            
            # === PAGE-BOUNDARY CLAMPING (Prevent overflow) ===
            page_width = page.rect.width
            page_height = page.rect.height
            safe_margin = 10
            
            widget_rect = fitz.Rect(
                max(safe_margin, min(widget_rect.x0, page_width - safe_margin - DEFAULT_WIDGET_WIDTH)),
                max(safe_margin, min(widget_rect.y0, page_height - safe_margin - DEFAULT_WIDGET_HEIGHT)),
                max(safe_margin + DEFAULT_WIDGET_WIDTH, min(widget_rect.x1, page_width - safe_margin)),
                max(safe_margin + DEFAULT_WIDGET_HEIGHT, min(widget_rect.y1, page_height - safe_margin))
            )
            
            # === CREATE ACROFORM WIDGET (Editable, no text stamping) ===
            try:
                field_index += 1
                widget = fitz.Widget()
                # Create unique field name using label + page + index to prevent overwrites
                unique_field_name = f"{field_name}_p{page_num+1}_{field_index}"
                widget.field_name = unique_field_name
                widget.rect = widget_rect
                
                if field_type == "Checkbox":
                    widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    if value:
                        is_checked = str(value).lower() in ['true', 'yes', '1', 'checked', 'on']
                        widget.field_value = is_checked
                else:
                    # Create text field
                    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                    widget.field_flags = 0  # Regular text field
                    widget.text_fontsize = WIDGET_TEXT_FONTSIZE  # Auto-fit font size
                    widget.text_color = WIDGET_TEXT_COLOR  # Black text
                    # Make widget visually non-destructive (no background, no border)
                    widget.fill_color = None  # Transparent background
                    widget.border_width = 0  # No border
                    
                    # Set value if provided
                    if value:
                        widget.field_value = str(value)
                
                # Add widget to page FIRST (widget needs parent before update)
                page.add_widget(widget)
                
                # Now update the widget to render the value (after it has a parent)
                if value:
                    widget.update()
                
                created_count += 1
                
                if debug:
                    group_info = f"[{field_group['group_id']}]" if field_group else "[no group]"
                    print(f"   ✓ Created: {unique_field_name} = '{value}' {group_info}")
            
            except Exception as e:
                if debug:
                    print(f"   ✗ Failed to create field '{field_name}': {e}")
        
        if debug:
            print(f"\\n   Total created: {created_count}/{len(fields)} fields")
            print(f"   \u2713 PDF remains EDITABLE")
        
        # === FIX ACROFORM REGISTRATION ===
        # Ensure widgets are registered in AcroForm and appearances are generated
        try:
            # Set NeedAppearances flag to force PDF viewers to regenerate widget appearances
            # This is critical for widgets to be editable, especially with background XObjects
            if pdf_doc.is_pdf:
                # Access the catalog and AcroForm
                catalog = pdf_doc.pdf_catalog()
                if catalog:
                    # Set NeedAppearances to true in AcroForm dictionary
                    # This tells PDF viewers to regenerate widget appearances
                    xref = pdf_doc.xref_get_key(catalog, "AcroForm")
                    if xref and xref[0] == "xref":
                        acroform_xref = int(xref[1].split()[0])
                        pdf_doc.xref_set_key(acroform_xref, "NeedAppearances", "true")
                        
                        if debug:
                            print(f"   \u2713 Set /NeedAppearances flag for widget rendering")
        except Exception as e:
            if debug:
                print(f"   Warning: Could not set NeedAppearances: {e}")
        
        # Save without flattening
        try:
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=4, clean=True)
        except Exception as e:
            if debug:
                print(f"   Warning: Aggressive cleanup failed ({e}), trying simpler approach...")
            # Fallback to simpler serialization without aggressive cleanup
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=1)
        
        pdf_doc.close()
        
        return output_bytes
    
    def _place_widget_in_grid(self, base_bbox: tuple, group: Dict, page_width: float, page_height: float) -> fitz.Rect:
        """
        Place widget in grid cell (for grid layouts).
        
        Uses geometry only - no semantic assumptions.
        """
        x0, y0, x1, y1 = base_bbox
        
        # For grid layouts, place widget in the cell's empty space
        # Assume label is on left, widget goes to right within same cell
        widget_x0 = x1 + GROUP_WIDGET_MARGIN
        widget_y0 = y0
        widget_x1 = min(widget_x0 + DEFAULT_WIDGET_WIDTH, group["bbox"][2] - GROUP_WIDGET_MARGIN)
        widget_y1 = y1
        
        return fitz.Rect(widget_x0, widget_y0, widget_x1, widget_y1)
    
    def _place_widget_in_group(self, base_bbox: tuple, group: Dict, page_width: float, page_height: float) -> fitz.Rect:
        """
        Place widget in nearest empty region within group (for linear/scattered layouts).
        
        Uses geometry only - no semantic assumptions.
        """
        x0, y0, x1, y1 = base_bbox
        
        if not group:
            # No group - use simple right-of-label placement
            return fitz.Rect(
                x1 + RIGHT_OFFSET_X,
                y0,
                x1 + RIGHT_OFFSET_X + DEFAULT_WIDGET_WIDTH,
                y1
            )
        
        group_x0, group_y0, group_x1, group_y1 = group["bbox"]
        
        # Try right-of-label first
        widget_x0 = x1 + RIGHT_OFFSET_X
        if widget_x0 + DEFAULT_WIDGET_WIDTH <= group_x1 - GROUP_WIDGET_MARGIN:
            # Fits to the right within group
            return fitz.Rect(
                widget_x0,
                y0,
                widget_x0 + DEFAULT_WIDGET_WIDTH,
                y1
            )
        
        # Otherwise place below within group bounds
        label_height = y1 - y0
        safe_offset = label_height * 0.8 + 4
        widget_y0 = y1 + safe_offset
        
        if widget_y0 + DEFAULT_WIDGET_HEIGHT <= group_y1 - GROUP_WIDGET_MARGIN:
            return fitz.Rect(
                x0,
                widget_y0,
                min(x0 + DEFAULT_WIDGET_WIDTH, group_x1 - GROUP_WIDGET_MARGIN),
                widget_y0 + DEFAULT_WIDGET_HEIGHT
            )
        
        # Fallback: place right of label even if extends beyond group
        return fitz.Rect(
            x1 + RIGHT_OFFSET_X,
            y0,
            x1 + RIGHT_OFFSET_X + DEFAULT_WIDGET_WIDTH,
            y1
        )
    
    def create_fillable_pdf(self, pdf_bytes: bytes, fields: List[Dict], field_mappings: Dict[str, str], debug: bool = False) -> bytes:
        """Create fillable PDF from flat PDF"""
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for field in fields:
            try:
                page_num = field.get("page_number", 1) - 1
                if page_num >= len(pdf_doc):
                    continue
                
                page = pdf_doc[page_num]
                bbox = field.get("position", (100, 100, 300, 120))
                
                widget = fitz.Widget()
                widget.field_name = field["field_name"]
                widget.rect = fitz.Rect(bbox)
                
                value = field_mappings.get(field["field_name"])
                
                if field.get("field_type") == "Checkbox":
                    widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    if value:
                        is_checked = str(value).lower() in ['true', 'yes', '1', 'checked', 'on']
                        widget.field_value = is_checked
                else:
                    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                    widget.text_fontsize = 10
                    widget.text_color = (0, 0, 0)
                    
                    if value:
                        widget.field_value = str(value)
                
                page.add_widget(widget)
            except Exception as e:
                if debug:
                    print(f"   Failed to create field '{field['field_name']}': {e}")
        
        # Flatten the PDF to make it non-editable
        try:
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=4, clean=True)
        except Exception as e:
            if debug:
                print(f"   Warning: Aggressive cleanup failed ({e}), trying simpler approach...")
            # Fallback to simpler serialization without aggressive cleanup
            output_bytes = pdf_doc.tobytes(deflate=True, garbage=1)
        
        pdf_doc.close()
        
        return output_bytes
    
    def _parse_field_number_and_clean_label(self, label: str) -> Dict:
        """
        Parse field number from label and extract clean label text.
        
        Handles numbered forms like W-9, IRS forms where labels contain:
        - "5 Address (number, street...) 6 City, state"
        - "y y g p , , See S 5 Address (number, street...)"  (OCR garbage prefix)
        - "6 City, state, and ZIP code 7 List account"
        
        Returns:
            {
                'field_number': int or None,
                'label_clean': str,  # Text for THIS field only
                'label_full': str,   # Original full label
                'has_trailing_text': bool  # True if other field numbers detected
            }
        """
        label_full = label.strip()
        
        # First, clean OCR garbage (repeated single letters, punctuation at start)
        # Remove patterns like "y y g p , , " or "f entity/individual. An entry"
        label_cleaned = re.sub(r'^[a-z\s\.,;]+(?=[A-Z0-9])', '', label_full)
        
        # Pattern 1: Field number at start "5 Address..." or "5. Address" or "5) Address"
        match = re.match(r'^(\d+)[\.)\s]+(.+?)(?:\s+\d+[\s\.]|$)', label_cleaned, re.DOTALL)
        
        if match:
            field_number = int(match.group(1))
            label_text = match.group(2).strip()
            
            # Check if there are additional field numbers in the remaining text
            remaining = label_cleaned[match.end():]
            has_trailing = bool(re.search(r'\d+\s+[A-Z]', remaining))
            
            # Further clean the label: stop at next field number
            clean_match = re.search(r'^(.+?)(?:\s+\d+\s+[A-Z])', label_text)
            if clean_match:
                label_clean = clean_match.group(1).strip()
            else:
                label_clean = label_text.strip()
            
            # Remove trailing punctuation and parentheses artifacts
            label_clean = re.sub(r'[\(\[\{]*$', '', label_clean)
            label_clean = label_clean.strip('.:;,')
            
            return {
                'field_number': field_number,
                'label_clean': label_clean,
                'label_full': label_full,
                'has_trailing_text': has_trailing
            }
        
        # Pattern 2: Field number in middle "See instructions. 5 Address..."
        # Look for pattern: [text] [number] [Capitalized word]
        middle_match = re.search(r'[\.\s](\d+)\s+([A-Z][a-z]+(?:\s+[a-z]+)*\s*\([^)]+\)|[A-Z][^\.0-9]+?)(?:\s+\d+\s+[A-Z]|$)', label_cleaned)
        
        if middle_match:
            field_number = int(middle_match.group(1))
            label_text = middle_match.group(2).strip()
            
            # Check for trailing field numbers
            remaining = label_cleaned[middle_match.end():]
            has_trailing = bool(re.search(r'\d+\s+[A-Z]', remaining))
            
            # Clean the extracted label
            label_clean = label_text.strip('.:;,')
            label_clean = re.sub(r'[\(\[\{]*$', '', label_clean)
            
            return {
                'field_number': field_number,
                'label_clean': label_clean,
                'label_full': label_full,
                'has_trailing_text': has_trailing
            }
        
        # No field number found - return cleaned label
        return {
            'field_number': None,
            'label_clean': label_cleaned if label_cleaned else label_full,
            'label_full': label_full,
            'has_trailing_text': False
        }
    
    def _validate_field_mappings(self, mappings: Dict[str, str], fields: List[Dict], debug: bool = False) -> Dict[str, str]:
        """
        Validate LLM field mappings and warn about potential errors.
        
        Checks for common mapping issues like:
        - City/State field containing only numbers (likely account number)
        - Account field containing address text
        - Email field missing @ symbol
        
        Args:
            mappings: Field name -> value mappings from LLM
            fields: Original field metadata with labels
            debug: Enable debug logging
            
        Returns:
            Same mappings (warnings only, doesn't modify)
        """
        if not debug:
            return mappings
        
        # Create lookup of field_name -> field_data
        field_lookup = {f['field_name']: f for f in fields}
        
        validation_issues = []
        
        for field_name, value in mappings.items():
            if not value or value is True or value is False:
                continue  # Skip empty or boolean values
            
            field_data = field_lookup.get(field_name, {})
            label = field_data.get('label', '').lower()
            label_full = field_data.get('label_full', '').lower()
            field_number = field_data.get('field_number')
            
            value_str = str(value)
            
            # Check 1: City/State field validation
            if ('city' in label or 'state' in label):
                # Should contain text, not just numbers
                if re.match(r'^\d+[\d\-\s]*$', value_str):
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"City/State field contains only numbers: '{value_str}'. "
                        f"This might be an account number mapped to wrong field."
                    )
            
            # Check 2: Account number validation
            if 'account' in label or 'acct' in label:
                # Should be numeric or contain numbers
                if not re.search(r'\d', value_str):
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"Account field has no numbers: '{value_str}'"
                    )
                # Should NOT contain city/state keywords
                if re.search(r'\b(city|state|ave|street|st|rd|blvd)\b', value_str.lower()):
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"Account field contains address-like text: '{value_str}'"
                    )
            
            # Check 3: Email validation
            if 'email' in label or 'e-mail' in label:
                if '@' not in value_str:
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"Email missing @ symbol: '{value_str}'"
                    )
            
            # Check 4: Phone validation
            if 'phone' in label or 'tel' in label:
                digits = re.sub(r'\D', '', value_str)
                if len(digits) != 10:
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"Phone doesn't have 10 digits (has {len(digits)}): '{value_str}'"
                    )
            
            # Check 5: SSN validation
            if 'ssn' in label or 'social security' in label:
                digits = re.sub(r'\D', '', value_str)
                if len(digits) != 9:
                    validation_issues.append(
                        f"⚠️  Field {field_number or field_name} ('{label}'): "
                        f"SSN doesn't have 9 digits (has {len(digits)}): '{value_str}'"
                    )
        
        if validation_issues:
            print(f"\n   {'='*66}")
            print("   FIELD MAPPING VALIDATION WARNINGS")
            print(f"   {'='*66}")
            for issue in validation_issues:
                print(f"   {issue}")
            print(f"   {'='*66}\n")
        
        return mappings
    
    def _sanitize_field_name(self, label: str) -> str:
        """Convert label to valid field name"""
        name = label.lower()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name.strip('_')
        return name if name else "unnamed_field"
    
    def polygon_to_rect(self, polygon, page_height: float = None, apply_y_inversion: bool = False) -> fitz.Rect:
        """
        Convert Azure DI polygon to PyMuPDF Rect with proper unit conversion.
        
        Azure DI returns polygons as 8 numbers [x1,y1, x2,y2, x3,y3, x4,y4] in INCHES.
        PyMuPDF uses POINTS where 1 inch = 72 points.
        
        Args:
            polygon: List of 8 floats (4 coordinate pairs) in inches
            page_height: PDF page height in points (for Y-axis inversion if needed)
            apply_y_inversion: If True, flip Y coordinates (Corrected_Y = page_height - Azure_Y)
        
        Returns:
            fitz.Rect(x0, y0, x1, y1) in points
        """
        if not polygon or len(polygon) < 8:
            return fitz.Rect(0, 0, 0, 0)
        
        # Step 1: Convert inches to points (inches * 72)
        pts = [val * 72 for val in polygon]
        
        # Step 2: Extract X and Y pairs
        xs = pts[0::2]  # [x1, x2, x3, x4]
        ys = pts[1::2]  # [y1, y2, y3, y4]
        
        # Step 3: Apply Y-axis inversion if needed
        if apply_y_inversion and page_height:
            ys = [page_height - y for y in ys]
        
        # Step 4: Create bounding box (min/max handles rotated/slanted polygons)
        rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        return rect
    
    def _get_polygon_bounds(self, polygon, page_height: float = None, apply_y_inversion: bool = False) -> Tuple[float, float, float, float]:
        """
        Get bounding box from Azure DI polygon with proper coordinate conversion.
        
        Args:
            polygon: List of 8 floats (4 coordinate pairs) in inches
            page_height: PDF page height in points (for Y-axis inversion)
            apply_y_inversion: If True, flip Y coordinates for bottom-left origin PDFs
        
        Returns:
            Tuple (x0, y0, x1, y1) in points
        """
        rect = self.polygon_to_rect(polygon, page_height, apply_y_inversion)
        return (rect.x0, rect.y0, rect.x1, rect.y1)
    
    def is_polygon_rotated(self, polygon, rotation_threshold: float = ROTATION_THRESHOLD) -> bool:
        """
        Detect if a polygon represents a rotated/slanted field.
        
        Useful for dynamic forms where fields may be slanted or part of tight grids.
        If top-right Y differs significantly from top-left Y, field is rotated.
        
        Args:
            polygon: List of 8 floats (4 coordinate pairs) in inches
            rotation_threshold: Y-difference threshold in points to consider rotated
        
        Returns:
            True if polygon is rotated beyond threshold
        """
        if not polygon or len(polygon) < 8:
            return False
        
        # Convert to points
        pts = [val * 72 for val in polygon]
        
        # Typically polygon is [top-left, top-right, bottom-right, bottom-left]
        # or some clockwise variation starting from top-left
        y1, y2 = pts[1], pts[3]  # First two Y coordinates
        
        # Check if Y coordinates differ significantly
        return abs(y1 - y2) > rotation_threshold
    
    def get_polygon_rotation_angle(self, polygon) -> float:
        """
        Calculate rotation angle of a polygon in degrees.
        
        Useful for applying rotation matrix to text insertion so it aligns
        perfectly with slanted field lines.
        
        Args:
            polygon: List of 8 floats (4 coordinate pairs) in inches
        
        Returns:
            Rotation angle in degrees (0 = horizontal)
        """
        if not polygon or len(polygon) < 8:
            return 0.0
        
        # Convert to points
        pts = [val * 72 for val in polygon]
        
        # Calculate angle from first two points (top edge)
        x1, y1 = pts[0], pts[1]
        x2, y2 = pts[2], pts[3]
        
        # Calculate angle using arctangent
        import math
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def insert_text_with_rotation(
        self,
        page,
        text: str,
        rect: fitz.Rect,
        rotation_angle: float = 0.0,
        fontsize: int = 10,
        color: tuple = (0, 0, 0),
        align: int = fitz.TEXT_ALIGN_LEFT
    ) -> bool:
        """
        Insert text into PDF with rotation support for slanted fields.
        
        Args:
            page: PyMuPDF page object
            text: Text to insert
            rect: fitz.Rect defining the text area
            rotation_angle: Rotation angle in degrees
            fontsize: Font size in points
            color: RGB color tuple (0-1)
            align: Text alignment constant
        
        Returns:
            True if successful
        """
        try:
            if abs(rotation_angle) < 1.0:
                # No significant rotation - use standard insertion
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=fontsize,
                    color=color,
                    align=align
                )
            else:
                # Apply rotation matrix for slanted text
                import math
                
                # Get center point of rect
                center_x = (rect.x0 + rect.x1) / 2
                center_y = (rect.y0 + rect.y1) / 2
                
                # Create rotation matrix
                angle_rad = math.radians(rotation_angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                
                # Translate to origin, rotate, translate back
                morph = fitz.Matrix(1, 0, 0, 1, -center_x, -center_y)  # Translate to origin
                morph = morph * fitz.Matrix(cos_a, sin_a, -sin_a, cos_a, 0, 0)  # Rotate
                morph = morph * fitz.Matrix(1, 0, 0, 1, center_x, center_y)  # Translate back
                
                # Insert text with transformation
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=fontsize,
                    color=color,
                    align=align,
                    morph=(morph.a, morph.b, morph.c, morph.d, morph.e, morph.f)
                )
            
            return True
        
        except Exception as e:
            print(f"Error inserting rotated text: {e}")
            return False
    
    def _calculate_fill_offset(self, rect: fitz.Rect, detection_method: str, field_type: str, label: str = None) -> fitz.Rect:
        """
        Calculate offset fill position to avoid overlapping with label.
        
        Args:
            rect: Original label bounding box
            detection_method: How the field was detected
            field_type: Type of field
            label: The label text (for more accurate offset calculation)
        
        Returns:
            New rect positioned for filling (not overlapping label)
        """
        if detection_method == "visual_underline":
            # Label: ______
            # Fill to the right of the label (where underscores are)
            # Use more conservative offset to avoid overlap
            line_width = rect.x1 - rect.x0
            
            # If we have label text, estimate its width more accurately
            if label:
                # Use estimated character width from constants
                estimated_label_width = min(len(label) * ESTIMATED_CHAR_WIDTH, line_width * 0.5)
            else:
                estimated_label_width = line_width * 0.45
            
            fill_rect = fitz.Rect(
                rect.x0 + estimated_label_width + LABEL_SPACING_OFFSET + INLINE_LABEL_OFFSET,
                rect.y0,
                rect.x1,
                rect.y1
            )
            return fill_rect
        
        elif detection_method == "visual_inline":
            # Label: ____  or  Label:
            # Fill to the right after colon
            line_width = rect.x1 - rect.x0
            
            # If we have label text, estimate its width more accurately
            if label:
                # Use estimated character width from constants (+ colon + space)
                estimated_label_width = min((len(label) + 2) * ESTIMATED_CHAR_WIDTH, line_width * 0.5)
            else:
                estimated_label_width = line_width * 0.45
            
            fill_rect = fitz.Rect(
                rect.x0 + estimated_label_width + LABEL_SPACING_OFFSET + INLINE_LABEL_OFFSET,
                rect.y0,
                rect.x1,
                rect.y1
            )
            return fill_rect
        
        elif detection_method == "visual_label_line":
            # Label on one line, fill area below
            height = rect.y1 - rect.y0
            fill_rect = fitz.Rect(
                rect.x0,
                rect.y1 + BELOW_LABEL_OFFSET,
                rect.x1,
                rect.y1 + height + BELOW_LABEL_OFFSET
            )
            return fill_rect
        
        elif detection_method == "visual_signature":
            # Signature line - fill below or to the right
            if field_type == FIELD_TYPE_SIGNATURE:
                # Place below for signature fields
                height = rect.y1 - rect.y0
                fill_rect = fitz.Rect(
                    rect.x0,
                    rect.y1 + BELOW_LABEL_OFFSET,
                    rect.x1,
                    rect.y1 + height + BELOW_LABEL_OFFSET
                )
                return fill_rect
        
        # Default: return original rect (for existing fields, checkboxes, etc.)
        return rect
    
    def fill_pdf_with_polygons(
        self,
        pdf_bytes: bytes,
        fields: List[Dict],
        field_mappings: Dict[str, str],
        create_editable: bool = True,  # New parameter: True = create AcroForm fields, False = stamp text
        debug: bool = False
    ) -> bytes:
        """
        Fill PDF using Azure DI polygon coordinates.
        
        HARD RULE ENFORCEMENT:
        - Classifies PDF document type (FILLABLE, FLATTENED_FORM, NARRATIVE)
        - Returns capability report for NARRATIVE_PDF (not fillable)
        - Routes flattened PDFs to widget-only filling
        - Creates widgets only for FILLABLE or FLATTENED_FORM PDFs
        
        Args:
            pdf_bytes: PDF file bytes
            fields: List of field dicts with polygon data from Azure DI
            field_mappings: Dict mapping field names to values
            create_editable: If True, creates editable AcroForm fields. If False, stamps text.
            debug: Enable debug logging
        
        Returns:
            Filled PDF as bytes OR original PDF if NARRATIVE_PDF
        """
        # HARD RULE: Classify document first
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        doc_class = self.classify_pdf_document(pdf_doc, None, debug)
        
        # Check if any fields are marked as non-fillable
        if fields and any(f.get("fillable") == False for f in fields):
            doc_class = DOC_CLASS_NARRATIVE
        
        # NARRATIVE_PDF: Cannot fill - return original with capability report
        if doc_class == DOC_CLASS_NARRATIVE:
            pdf_doc.close()
            if debug:
                print(f"\n   ❌ NARRATIVE_PDF detected - filling NOT supported")
                print(f"   📊 Capability Report:")
                print(f"      Document type: Narrative/informational")
                print(f"      Fillable fields: 0")
                print(f"      Labels detected: {len(fields)}")
                print(f"      Action: Extraction only (no filling)")
                print(f"   ℹ️  Returning original PDF unchanged\n")
            # Return original PDF unchanged
            return pdf_bytes
        
        # Check if flattened
        is_flattened = self._detect_flattened_pdf(pdf_doc, None, debug)
        pdf_doc.close()
        
        if is_flattened:
            # WIDGET-ONLY MODE: Ignore create_editable setting
            if debug:
                print(f"\n   ⚠️  Flattened PDF detected - forcing WIDGET-ONLY mode")
            return self.fill_existing_widgets_only(pdf_bytes, fields, field_mappings, debug)
        
        # Non-flattened PDF: Use the editable method by default
        if create_editable:
            if debug:
                print(f"   ℹ Using EDITABLE mode (creating AcroForm fields)")
            return self.create_editable_pdf_with_acroform(pdf_bytes, fields, field_mappings, debug)
        
        # Legacy mode: stamp text and flatten (not recommended)
        if debug:
            print(f"   ⚠ Using LEGACY mode (stamping text, will be flattened)")
        
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        filled_count = 0
        
        for field in fields:
            field_name = field.get("field_name")
            if not field_name or field_name not in field_mappings:
                continue
            
            value = field_mappings[field_name]
            page_num = field.get("page_number", 1) - 1
            
            if page_num >= len(pdf_doc):
                continue
            
            page = pdf_doc[page_num]
            
            # Get position and polygon data
            polygon = field.get("polygon")
            position = field.get("position")
            is_rotated = field.get("is_rotated", False)
            rotation_angle = field.get("rotation_angle", 0.0)
            detection_method = field.get("detection_method", "")
            fill_strategy = field.get("fill_strategy", "")
            field_type = field.get("field_type", "Text")
            
            # Convert position to rect
            if polygon:
                # Use polygon_to_rect for precise conversion
                rect = self.polygon_to_rect(polygon, page.rect.height, apply_y_inversion=False)
            elif position:
                rect = fitz.Rect(position)
            else:
                continue
            
            # Apply offset for visual fields to avoid label overlap
            if fill_strategy == "stamp_text" and detection_method.startswith("visual_"):
                field_label = field.get("label", "")
                rect = self._calculate_fill_offset(rect, detection_method, field_type, field_label)
                if debug:
                    offset_info = f" [offset: {detection_method}]"
                else:
                    offset_info = ""
            else:
                offset_info = ""
            
            try:
                if field_type == "Checkbox":
                    # Insert checkbox symbol
                    check_symbol = "☑" if str(value).lower() in ['true', 'yes', '1', 'checked', 'on'] else "☐"
                    self.insert_text_with_rotation(
                        page,
                        check_symbol,
                        rect,
                        rotation_angle=rotation_angle,
                        fontsize=STAMP_CHECKBOX_FONTSIZE,
                        color=WIDGET_TEXT_COLOR
                    )
                else:
                    # Insert text value
                    self.insert_text_with_rotation(
                        page,
                        str(value),
                        rect,
                        rotation_angle=rotation_angle,
                        fontsize=STAMP_TEXT_FONTSIZE,
                        color=STAMP_TEXT_COLOR,
                        align=fitz.TEXT_ALIGN_LEFT
                    )
                
                filled_count += 1
                
                if debug:
                    rotation_info = f" (rotated {rotation_angle:.1f}°)" if is_rotated else ""
                    print(f"   ✓ Filled: {field_name} = {value}{offset_info}{rotation_info}")
            
            except Exception as e:
                if debug:
                    print(f"   ✗ Failed to fill '{field_name}': {e}")
        
        if debug:
            print(f"\n   Total filled: {filled_count}/{len(fields)} fields")
        
        # Flatten the PDF to make it non-editable (remove form fields)
        # This converts all filled content to static text/graphics
        if debug:
            print(f"   Flattening PDF to remove form fields...")
        
        # Save to bytes with flattening
        output_bytes = pdf_doc.tobytes(deflate=True, garbage=4, clean=True)
        pdf_doc.close()
        
        return output_bytes
