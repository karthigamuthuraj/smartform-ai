"""
PDF Quality Analyzer - Automatically determines optimal CommonForms settings
"""

import numpy as np
from pypdf import PdfReader
from pdf2image import convert_from_path
from typing import Dict, Tuple


class PDFQualityAnalyzer:
    """Analyzes PDF quality and recommends CommonForms settings"""
    
    @staticmethod
    def analyze_quality(pdf_path: str) -> Dict:
        """
        Analyze PDF quality and return recommended settings
        
        Returns:
            dict: {
                'quality': str,  # 'good', 'medium', 'poor'
                'sharpness': float,
                'variance': float,
                'is_scanned': bool,
                'recommended_image_size': int,
                'recommended_confidence': float,
                'recommended_model': str
            }
        """
        try:
            # Check if PDF has existing form fields
            has_fields = PDFQualityAnalyzer._check_has_fields(pdf_path)
            
            # Get page count and dimensions
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            page_width = float(reader.pages[0].mediabox.width)
            page_height = float(reader.pages[0].mediabox.height)
            
            # Analyze image quality and text density
            sharpness, variance, text_density = PDFQualityAnalyzer._calculate_image_metrics(pdf_path)
            
            # Base quality determination
            if sharpness < 10:
                base_quality = 'poor'
                base_image_size = 2400
                confidence = 0.15
            elif sharpness < 30:
                base_quality = 'medium'
                base_image_size = 2048
                confidence = 0.2
            else:
                base_quality = 'good'
                base_image_size = 1600
                confidence = 0.25
            
            # Determine if scanned
            is_scanned = variance < 1000 or sharpness < 15
            
            # ENHANCED RESOLUTION LOGIC
            # Start with base, then apply adjustments
            image_size = base_image_size
            adjustments = []
            
            # Factor 1: Text density - high density = more fields = need higher resolution
            if text_density > 0.25:  # Dense form (>25% coverage)
                image_size = max(image_size, 2400)
                adjustments.append("high text density detected (+resolution)")
            elif text_density > 0.15:  # Medium density
                image_size = max(image_size, 2048)
                adjustments.append("medium text density")
            
            # Factor 2: Scanned documents need higher resolution
            if is_scanned:
                image_size = max(image_size, 2400)
                adjustments.append("scanned document (+resolution)")
            
            # Factor 3: Page size - larger pages might have smaller text
            page_area = page_width * page_height
            if page_area > 500000:  # Large page (e.g., legal size)
                image_size = max(image_size, 2400)
                adjustments.append("large page size (+resolution)")
            
            # Factor 4: Low sharpness with high text density = challenging form
            if sharpness < 20 and text_density > 0.15:
                image_size = max(image_size, 2800)
                adjustments.append("low sharpness + dense text (+resolution)")
            
            # Factor 5: For multi-page forms, be conservative
            if num_pages > 1 and image_size >= 2400:
                # Keep high resolution for accuracy, page-by-page will handle memory
                adjustments.append(f"{num_pages} pages (page-by-page processing)")
            
            # Factor 6: Safety margin - if borderline, go higher
            # Better to use more resources than miss fields
            if base_quality == 'medium' and image_size < 2400:
                image_size = 2400
                adjustments.append("conservative boost for reliability")
            
            quality = base_quality
            adjustment_note = "; ".join(adjustments) if adjustments else "using base settings"
            
            return {
                'quality': quality,
                'sharpness': round(sharpness, 2),
                'variance': round(variance, 0),
                'text_density': round(text_density, 3),
                'is_scanned': is_scanned,
                'has_existing_fields': has_fields,
                'num_pages': num_pages,
                'page_size': f"{int(page_width)}x{int(page_height)}",
                'recommended_image_size': image_size,
                'recommended_confidence': confidence,
                'recommended_model': 'FFDNet-L',
                'quality_description': PDFQualityAnalyzer._get_quality_description(quality, is_scanned),
                'resolution_adjustments': adjustment_note
            }
            
        except Exception as e:
            # Fallback to safe defaults if analysis fails
            return {
                'quality': 'medium',
                'sharpness': 0,
                'variance': 0,
                'text_density': 0,
                'is_scanned': False,
                'has_existing_fields': False,
                'num_pages': 1,
                'page_size': 'unknown',
                'recommended_image_size': 2400,  # Conservative fallback
                'recommended_confidence': 0.2,
                'recommended_model': 'FFDNet-L',
                'quality_description': f'Analysis failed, using defaults: {str(e)}',
                'resolution_adjustments': 'analysis failed',
                'error': str(e)
            }
    
    @staticmethod
    def _check_has_fields(pdf_path: str) -> bool:
        """Check if PDF already has form fields"""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                if '/Annots' in page:
                    return True
            return False
        except:
            return False
    
    @staticmethod
    def _calculate_image_metrics(pdf_path: str) -> Tuple[float, float, float]:
        """
        Calculate sharpness, variance, and text density metrics
        
        Returns:
            Tuple[float, float, float]: (sharpness_score, pixel_variance, text_density)
        """
        try:
            # Convert first page to image at medium resolution
            images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
            img = images[0]
            
            # Convert to grayscale numpy array
            gray = np.array(img.convert('L'))
            
            # Calculate sharpness (gradient magnitude)
            gradient_x = np.abs(np.diff(gray, axis=1)).mean()
            gradient_y = np.abs(np.diff(gray, axis=0)).mean()
            sharpness = (gradient_x + gradient_y) / 2
            
            # Calculate pixel variance
            variance = float(np.var(gray))
            
            # Calculate text density (% of non-white pixels)
            # Threshold at 240 to consider as "content" vs background
            content_pixels = np.sum(gray < 240)
            total_pixels = gray.size
            text_density = content_pixels / total_pixels
            
            return sharpness, variance, text_density
            
        except Exception as e:
            # If image conversion fails, return medium defaults
            return 20.0, 1500.0, 0.15
    
    @staticmethod
    def _get_quality_description(quality: str, is_scanned: bool) -> str:
        """Get human-readable quality description"""
        descriptions = {
            'good': 'Clean digital PDF - using optimal settings for fast processing',
            'medium': 'Scanned or compressed PDF - using balanced settings',
            'poor': 'Low quality scan - using maximum quality settings for best detection'
        }
        
        desc = descriptions.get(quality, 'Unknown quality')
        
        if is_scanned:
            desc += ' (detected as scanned document)'
        
        return desc
