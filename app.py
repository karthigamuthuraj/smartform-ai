"""
Flask API for Production Document Filler (PDF-only demo build)
Handles PDF document filling through web interface
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from production_document_filler import ProductionDocumentFillingSystem
from logger import FormFillerLogger
from pdf_quality_analyzer import PDFQualityAnalyzer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize logger
logger = FormFillerLogger()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize the document filling system (PDF only)
document_system = ProductionDocumentFillingSystem(
    doc_intelligence_endpoint=os.getenv('AZURE_DOC_INTELLIGENCE_ENDPOINT'),
    doc_intelligence_key=os.getenv('AZURE_DOC_INTELLIGENCE_KEY'),
    openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview'),
    logger=logger
)

pdf_handler = document_system.pdf_handler

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

# =============================================================================
# WORKFLOW:
# 1. User uploads PDF → /api/upload (returns filepath)
# 2. User enters data and clicks Submit → /api/process-and-fill
#
# SEQUENTIAL PIPELINE IN /api/process-and-fill:
# → STEP 1: CommonForms conversion (if needed)
# → STEP 2: Azure DI field extraction (from converted PDF)
# → STEP 3: Field mapping (user data → extracted fields)
# → STEP 4: Document filling (with mapped data)
# → STEP 5: Return filled document
#
# Each step validates success before proceeding to next step.
# =============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload"""
    logger.log_section("FILE UPLOAD REQUEST")
    logger.log(f"Request method: {request.method}")
    logger.log(f"Request files: {list(request.files.keys())}")

    try:
        if 'file' not in request.files:
            logger.log("ERROR: No file in request!")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        logger.log(f"File received: {file.filename}")

        if file.filename == '':
            logger.log("ERROR: Empty filename!")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.log(f"ERROR: Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Only PDF allowed in this demo'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        logger.log(f"✅ File saved: {filepath}")
        logger.log(f"   Size: {os.path.getsize(filepath)} bytes")

        response_data = {
            'success': True,
            'filename': unique_filename,
            'original_filename': filename,
            'file_type': 'pdf',
            'filepath': unique_filename  # Return just the filename, not full path
        }

        logger.log(f"✅ Upload successful!")
        logger.log(f"   Response: {response_data}")

        return jsonify(response_data)

    except Exception as e:
        logger.log(f"❌ Upload exception: {str(e)}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/convert-pdf', methods=['POST'])
def convert_pdf():
    """Convert PDF to fillable format using CommonForms AI (background task for PDF upload)"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({'error': 'Missing filepath'}), 400

        logger.log(f"=== /api/convert-pdf called ===")
        logger.log(f"Filepath: {filepath}")

        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)

        if not os.path.exists(full_filepath):
            return jsonify({'error': 'File not found'}), 404

        with open(full_filepath, 'rb') as f:
            file_bytes = f.read()

        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        has_fields = False
        field_count = 0
        num_pages = len(doc)

        for page in doc:
            widgets = list(page.widgets())
            if widgets:
                has_fields = True
                field_count += len(widgets)

        doc.close()

        logger.log(f"PDF Analysis:")
        logger.log(f"  • Pages: {num_pages}")
        logger.log(f"  • Fillable fields detected: {field_count}")

        if has_fields:
            logger.log(f"✓ PDF already has {field_count} fillable fields - no conversion needed")
            return jsonify({
                'success': True,
                'message': f'PDF already has {field_count} fillable fields',
                'conversion_needed': False,
                'field_count': field_count
            })

        logger.log("Converting PDF with CommonForms AI...")

        logger.log("Analyzing PDF quality to optimize conversion settings...")
        quality_info = PDFQualityAnalyzer.analyze_quality(full_filepath)

        logger.log(f"  Quality: {quality_info['quality'].upper()}")
        logger.log(f"  Sharpness: {quality_info['sharpness']}")
        logger.log(f"  Text Density: {quality_info['text_density']}")
        logger.log(f"  Is Scanned: {quality_info['is_scanned']}")
        logger.log(f"  Recommended Image Size: {quality_info['recommended_image_size']}")
        logger.log(f"  Recommended Confidence: {quality_info['recommended_confidence']}")
        logger.log(f"  Adjustments: {quality_info['resolution_adjustments']}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(full_filepath))[0]
        converted_filename = f"{base_name}_fillable.pdf"
        converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)

        success = process_pdf_page_by_page(
            input_filepath=full_filepath,
            output_filepath=converted_filepath,
            confidence=quality_info['recommended_confidence'],
            image_size=quality_info['recommended_image_size'],
            model=quality_info['recommended_model'],
            device='cpu',
            multiline=True,
            keep_existing=False,
            logger=logger
        )

        if success and os.path.exists(converted_filepath):
            logger.log(f"✓ Converted PDF saved as new file:")
            logger.log(f"  Original: {filepath}")
            logger.log(f"  Converted: {converted_filename}")
            logger.log("✓ PDF conversion complete!")

            return jsonify({
                'success': True,
                'message': 'PDF converted successfully',
                'conversion_needed': True,
                'original_filepath': filepath,
                'converted_filepath': converted_filename
            })
        else:
            logger.log("❌ PDF conversion failed")
            if os.path.exists(converted_filepath):
                os.remove(converted_filepath)
            return jsonify({'error': 'PDF conversion failed'}), 500

    except Exception as e:
        logger.log(f"❌ Convert PDF exception: {str(e)}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500


@app.route('/api/process-and-fill', methods=['POST'])
def process_and_fill():
    """
    Sequential Pipeline: User submits → Process everything in order

    STRICT ORDER (do not reorder):
    1. CommonForms conversion (if needed)
    2. Azure DI field extraction (on converted PDF)
    3. Field mapping (user data → fields)
    4. PDF filling (mapped data → converted PDF)
    5. Return filled PDF
    """
    try:
        data = request.json
        filepath = data.get('filepath')
        natural_language_input = data.get('natural_language_input', '')

        logger.log_section("SEQUENTIAL PIPELINE: PDF DOCUMENT")
        logger.log(f"Original file: {filepath}")
        logger.log(f"Pipeline: CommonForms → Azure DI → Mapping → Filling")

        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)

        if not os.path.exists(full_filepath):
            return jsonify({'error': 'File not found'}), 404

        with open(full_filepath, 'rb') as f:
            file_bytes = f.read()

        # STEP 1: CommonForms Conversion
        logger.log("=" * 80)
        logger.log("STEP 1: COMMONFORMS CONVERSION")
        logger.log("=" * 80)

        import fitz
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        has_fields = False
        field_count = 0

        for page in pdf_doc:
            widgets = list(page.widgets())
            if widgets:
                has_fields = True
                field_count += len(widgets)

        pdf_doc.close()

        logger.log(f"PDF Analysis:")
        logger.log(f"  • Pages: {len(fitz.open(stream=file_bytes, filetype='pdf'))}")
        logger.log(f"  • Existing fillable fields: {field_count}")

        if has_fields:
            logger.log(f"✓ PDF already has {field_count} fillable fields - skipping conversion")
            converted_filepath = full_filepath
            converted_bytes = file_bytes
        else:
            logger.log(f"⚙ Converting static PDF to fillable PDF with CommonForms...")

            quality_info = PDFQualityAnalyzer.analyze_quality(full_filepath)

            logger.log(f"  Quality Analysis:")
            logger.log(f"    • Quality: {quality_info['quality'].upper()}")
            logger.log(f"    • Sharpness: {quality_info['sharpness']}")
            logger.log(f"    • Text Density: {quality_info['text_density']}")
            logger.log(f"    • Recommended Image Size: {quality_info['recommended_image_size']}")
            logger.log(f"    • Recommended Confidence: {quality_info['recommended_confidence']}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            converted_filename = f"{base_name}_fillable_{timestamp}.pdf"
            converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)

            success = process_pdf_page_by_page(
                input_filepath=full_filepath,
                output_filepath=converted_filepath,
                confidence=quality_info['recommended_confidence'],
                image_size=quality_info['recommended_image_size'],
                model=quality_info['recommended_model'],
                device='cpu',
                multiline=True,
                keep_existing=False,
                logger=logger
            )

            if not success or not os.path.exists(converted_filepath):
                logger.log("❌ STEP 1 FAILED: CommonForms conversion failed")
                return jsonify({'error': 'PDF conversion failed'}), 500

            with open(converted_filepath, 'rb') as f:
                converted_bytes = f.read()

            logger.log(f"✓ STEP 1 COMPLETE: Converted PDF saved as {converted_filename}")

        # STEP 2: Field Extraction (Azure DI)
        logger.log("=" * 80)
        logger.log("STEP 2: AZURE DOCUMENT INTELLIGENCE EXTRACTION")
        logger.log("=" * 80)
        logger.log(f"  Extracting fields from converted fillable PDF...")

        pdf_doc = fitz.open(stream=converted_bytes, filetype="pdf")

        existing_fields = pdf_handler.extract_existing_fields(pdf_doc)

        if existing_fields:
            logger.log(f"  ✓ Found {len(existing_fields)} existing fillable fields in converted PDF")

            enhanced_fields = pdf_handler.enhance_fields_with_azure_di(
                pdf_bytes=converted_bytes,
                existing_fields=existing_fields,
                pdf_doc=pdf_doc,
                logger=logger,
                debug=True
            )

            fields = pdf_handler.enhance_labels_with_llm(
                pdf_bytes=converted_bytes,
                existing_fields=enhanced_fields,
                pdf_doc=pdf_doc,
                logger=logger,
                debug=True
            )
        else:
            logger.log("  No existing fields, using Azure DI for full detection...")
            fields = pdf_handler.detect_fields_with_azure_di(converted_bytes, pdf_doc, debug=True)

        try:
            pdf_doc.close()
        except Exception:
            pass

        if not fields:
            logger.log("❌ STEP 2 FAILED: No fields extracted")
            return jsonify({'error': 'No fields could be extracted'}), 400

        logger.log(f"✓ STEP 2 COMPLETE: Extracted {len(fields)} fields")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        extracted_fields_filename = f"{base_name}_extracted_fields_{timestamp}.json"
        extracted_fields_filepath = os.path.join(app.config['OUTPUT_FOLDER'], extracted_fields_filename)

        with open(extracted_fields_filepath, 'w', encoding='utf-8') as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        logger.log(f"  ✓ Saved: {extracted_fields_filename}")

        # STEP 3: Field Mapping
        logger.log("=" * 80)
        logger.log("STEP 3: FIELD MAPPING")
        logger.log("=" * 80)
        logger.log(f"  Mapping {len(fields)} fields to user input...")

        field_mappings = pdf_handler.map_data_to_fields(fields, natural_language_input, debug=True)

        if not field_mappings:
            logger.log("❌ STEP 3 FAILED: No mappings generated")
            return jsonify({'error': 'Field mapping failed'}), 400

        mapped_count = len([v for v in field_mappings.values() if v])
        logger.log(f"✓ STEP 3 COMPLETE: Mapped {mapped_count} of {len(fields)} fields")

        field_mappings_filename = f"{base_name}_field_mappings_{timestamp}.json"
        field_mappings_filepath = os.path.join(app.config['OUTPUT_FOLDER'], field_mappings_filename)
        with open(field_mappings_filepath, 'w', encoding='utf-8') as f:
            json.dump(field_mappings, f, indent=2, ensure_ascii=False)
        logger.log(f"  ✓ Saved: {field_mappings_filename}")

        # STEP 4: Fill Document
        logger.log("=" * 80)
        logger.log("STEP 4: FILL DOCUMENT")
        logger.log("=" * 80)

        filled_mappings = {k: v for k, v in field_mappings.items() if v}
        if filled_mappings:
            sample_count = min(5, len(filled_mappings))
            logger.log(f"  Sample mappings (showing first {sample_count}):")
            for field_name, value in list(filled_mappings.items())[:sample_count]:
                logger.log(f"    • {field_name} = {value}")
            if len(filled_mappings) > sample_count:
                logger.log(f"    ... and {len(filled_mappings) - sample_count} more fields")

        filled_filename = f"{base_name}_filled_{timestamp}.pdf"
        filled_filepath = os.path.join(app.config['OUTPUT_FOLDER'], filled_filename)

        pdf_doc = fitz.open(stream=converted_bytes, filetype="pdf")
        existing_fields = pdf_handler.extract_existing_fields(pdf_doc)

        if existing_fields:
            logger.log(f"  Filling {len(existing_fields)} PDF form fields...")
            filled_bytes = pdf_handler.fill_existing_pdf(pdf_doc, field_mappings, debug=True)
            try:
                pdf_doc.close()
            except Exception:
                pass
        else:
            try:
                pdf_doc.close()
            except Exception:
                pass

            has_polygons = any(field.get("polygon") for field in fields)
            if has_polygons:
                logger.log(f"  Creating fillable PDF using polygon coordinates...")
                filled_bytes = pdf_handler.fill_pdf_with_polygons(converted_bytes, fields, field_mappings, debug=True)
            else:
                logger.log(f"  Creating basic fillable PDF...")
                filled_bytes = pdf_handler.create_fillable_pdf(converted_bytes, fields, field_mappings, debug=True)

        with open(filled_filepath, 'wb') as f:
            f.write(filled_bytes)

        if not os.path.exists(filled_filepath):
            logger.log("❌ STEP 4 FAILED: Filled document not saved")
            return jsonify({'error': 'Failed to save filled document'}), 500

        logger.log(f"✓ STEP 4 COMPLETE: Filled document saved as {filled_filename}")

        # STEP 5: Return Success Response
        logger.log("=" * 80)
        logger.log("🎉 PIPELINE COMPLETE!")
        logger.log("=" * 80)
        logger.log(f"  Final output: {filled_filename}")
        logger.log(f"    • Extracted fields: {len(fields)}")
        logger.log(f"    • Mapped fields: {mapped_count}")

        return jsonify({
            'success': True,
            'output_filename': filled_filename,
            'total_fields': len(fields),
            'filled_count': mapped_count,
            'extracted_fields_json': extracted_fields_filename,
            'field_mappings_json': field_mappings_filename,
            'field_mappings': field_mappings
        })

    except Exception as e:
        logger.log("=" * 80)
        logger.log("❌ PIPELINE FAILED")
        logger.log("=" * 80)
        logger.log(f"Error: {str(e)}")
        import traceback
        logger.log(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Pipeline failed: {str(e)}'}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download filled document"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


def process_pdf_page_by_page(input_filepath, output_filepath, confidence, image_size, model, device, multiline, keep_existing, logger):
    """
    Optimized batch processing for large PDFs at high resolution.

    Strategy:
    - Process in batches of 6 pages (1-6, 7-12, 13-18, etc.)
    - Final batch handles remaining pages (e.g., if 14 pages: 1-6, 7-12, 13-14)
    - Much faster than page-by-page processing
    """
    from pypdf import PdfReader, PdfWriter
    import subprocess

    try:
        reader = PdfReader(input_filepath)
        num_pages = len(reader.pages)

        BATCH_SIZE = 6
        num_batches = (num_pages + BATCH_SIZE - 1) // BATCH_SIZE

        logger.log(f"📄 Processing {num_pages} pages in {num_batches} batch(es):")
        logger.log(f"   • Batch size: {BATCH_SIZE} pages")
        logger.log(f"   • Strategy: Optimized bulk processing (faster than page-by-page)")

        processed_pages = []
        temp_dir = app.config['UPLOAD_FOLDER']

        for batch_num in range(num_batches):
            start_page = batch_num * BATCH_SIZE
            end_page = min(start_page + BATCH_SIZE, num_pages)
            batch_page_count = end_page - start_page

            logger.log(f"\n📦 Batch {batch_num + 1}/{num_batches}: Processing pages {start_page + 1}-{end_page}...")

            batch_writer = PdfWriter()
            for i in range(start_page, end_page):
                batch_writer.add_page(reader.pages[i])

            temp_batch_input = os.path.join(temp_dir, f"temp_batch_{start_page+1}_{end_page}_{os.getpid()}.pdf")
            temp_batch_output = os.path.join(temp_dir, f"temp_batch_{start_page+1}_{end_page}_{os.getpid()}_filled.pdf")

            with open(temp_batch_input, "wb") as f:
                batch_writer.write(f)

            try:
                cmd = ['commonforms', temp_batch_input, temp_batch_output,
                       '--confidence', str(confidence),
                       '--image-size', str(image_size),
                       '--model', model,
                       '--device', device]

                if multiline:
                    cmd.append('--multiline')
                if keep_existing:
                    cmd.append('--keep-existing-fields')

                timeout = max(60 * batch_page_count, 120)

                result = subprocess.run(cmd, capture_output=True, timeout=timeout)

                if result.returncode == 0 and os.path.exists(temp_batch_output):
                    logger.log(f"   ✓ Batch {batch_num + 1} processed successfully ({batch_page_count} pages)")
                    processed_pages.append(('batch', temp_batch_output, batch_page_count, start_page))
                else:
                    logger.log(f"   ⚠ Batch {batch_num + 1} failed, using original pages")
                    processed_pages.append(('batch_failed', temp_batch_input, batch_page_count, start_page))

            except subprocess.TimeoutExpired:
                logger.log(f"   ⚠ Batch {batch_num + 1} timeout, using original pages")
                processed_pages.append(('batch_failed', temp_batch_input, batch_page_count, start_page))
            except Exception as e:
                logger.log(f"   ⚠ Batch {batch_num + 1} error: {e}, using original pages")
                processed_pages.append(('batch_failed', temp_batch_input, batch_page_count, start_page))

        logger.log("\n📑 Merging processed batches...")
        final_writer = PdfWriter()

        import fitz  # For field renaming
        global_page_counter = 0

        for proc_type, page_file, page_count, start_page_idx in processed_pages:
            if os.path.exists(page_file):
                try:
                    page_doc = fitz.open(page_file)

                    logger.log(f"    Processing batch: pages {start_page_idx + 1}-{start_page_idx + page_count}")

                    for page_idx, page in enumerate(page_doc):
                        global_page_counter += 1
                        widgets = list(page.widgets())

                        if widgets and page_idx == 0:
                            logger.log(f"      Renaming fields: adding _page{global_page_counter}+ suffix")

                        for widget in widgets:
                            old_name = widget.field_name
                            actual_page_num = start_page_idx + page_idx + 1
                            new_name = f"{old_name}_page{actual_page_num}"
                            widget.field_name = new_name
                            widget.update()

                    renamed_temp = os.path.join(temp_dir, f"renamed_batch_{start_page_idx+1}_{os.getpid()}.pdf")
                    page_doc.save(renamed_temp)
                    page_doc.close()

                    segment_reader = PdfReader(renamed_temp)
                    for page in segment_reader.pages:
                        final_writer.add_page(page)

                    os.remove(renamed_temp)
                    if not proc_type.endswith('_failed'):
                        os.remove(page_file)

                except Exception as e:
                    logger.log(f"    ⚠ Warning: Field renaming failed for batch {start_page_idx+1}: {e}")
                    if os.path.exists(page_file):
                        segment_reader = PdfReader(page_file)
                        for page in segment_reader.pages:
                            final_writer.add_page(page)
                        if not proc_type.endswith('_failed'):
                            os.remove(page_file)

        with open(output_filepath, "wb") as f:
            final_writer.write(f)

        logger.log(f"✓ Optimized processing complete! ({num_pages} pages merged)")
        return True

    except Exception as e:
        logger.log(f"ERROR in optimized processing: {str(e)}")
        import traceback
        logger.log(f"Traceback: {traceback.format_exc()}")
        return False


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'handlers': {
            'pdf': pdf_handler is not None
        }
    })

if __name__ == '__main__':
    print("🚀 Form Filler PDF Demo API Starting...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"📝 Log file: {logger.get_log_file()}")
    print("🌐 Access at: http://localhost:5000")
    logger.log("Flask server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
