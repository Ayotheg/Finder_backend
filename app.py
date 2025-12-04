from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import base64
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback
from PIL import Image
import io

load_dotenv()

app = Flask(__name__, static_folder='public')

# Better CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Roboflow Configuration
ROBOFLOW_CONFIG = {
    'api_url': 'https://serverless.roboflow.com',
    'api_key': os.getenv("ROBOFLOW_API_KEY"),
    'workspace': 'the-bunker',
    'workflow_id': 'sam3-with-prompts'
}

# Validate configuration at startup
if not ROBOFLOW_CONFIG['api_key']:
    print("⚠️  WARNING: ROBOFLOW_API_KEY not set in environment!")
else:
    print(f"✓ Roboflow API Key configured (ending with: ...{ROBOFLOW_CONFIG['api_key'][-4:]})")
    print(f"✓ Workspace: {ROBOFLOW_CONFIG['workspace']}")
    print(f"✓ Workflow ID: {ROBOFLOW_CONFIG['workflow_id']}")

# Create uploads folder if it doesn't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(file_path, max_size_mb=5, quality=85):
    """Compress image if it's too large - important for mobile photos"""
    try:
        img = Image.open(file_path)
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize if image is very large (common with mobile photos)
        max_dimension = 1920  # Max width or height
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  Image resized to: {new_size}")
        
        # Save with compression
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        compressed_data = output.getvalue()
        print(f"  Compressed size: {len(compressed_data) / 1024:.2f} KB")
        return compressed_data
        
    except Exception as e:
        print(f"  Image compression failed: {e}")
        # Return original if compression fails
        with open(file_path, 'rb') as f:
            return f.read()


@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        'name': 'Finder AI Backend',
        'version': '2.0',
        'endpoints': {
            'health': '/api/health',
            'analyze': '/api/analyze'
        },
        'features': ['SAM 3 Text Prompts', 'Smart Object Detection']
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Finder AI Backend is running',
        'roboflow_configured': bool(ROBOFLOW_CONFIG['api_key'])
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Main endpoint for image analysis with SAM 3 text prompts"""
    uploaded_file_path = None
    
    try:
        print("\n" + "="*50)
        print("NEW REQUEST RECEIVED")
        print("="*50)
        
        # Check if image file is present
        if 'image' not in request.files:
            print("❌ No image file in request")
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400

        # Get prompt from form data
        prompt = request.form.get('prompt', '').strip()
        
        # Require a prompt for SAM 3
        if not prompt:
            print("⚠️  No prompt provided - using default")
            prompt = "all objects"  # Default prompt

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        
        file_size = os.path.getsize(uploaded_file_path)
        print(f"📁 File saved: {filename}")
        print(f"📊 Original size: {file_size / 1024:.2f} KB")
        print(f"🔍 User prompt: '{prompt}'")

        # Compress image (important for mobile photos)
        print("🔄 Compressing image...")
        compressed_image = compress_image(uploaded_file_path)
        encoded_image = base64.b64encode(compressed_image).decode('utf-8')
        print(f"✓ Image encoded (base64 length: {len(encoded_image)} chars)")

        # Prepare Roboflow API request with SAM 3 text prompts
        roboflow_url = f"{ROBOFLOW_CONFIG['api_url']}/{ROBOFLOW_CONFIG['workspace']}/workflows/{ROBOFLOW_CONFIG['workflow_id']}"
        
        # UPDATED: Include prompts as ARRAY for SAM 3
        # SAM 3 expects prompts as a list/array, not a single string
        prompts_array = [p.strip() for p in prompt.split(',')] if ',' in prompt else [prompt]
        
        # Try simpler payload structure first
        payload = {
            'api_key': ROBOFLOW_CONFIG['api_key'],
            'inputs': {
                'image': encoded_image,  # Try without type/value wrapper
                'prompts': prompts_array
            }
        }
        
        print(f"📦 Payload structure:")
        print(f"   - image: {len(encoded_image)} chars (base64)")
        print(f"   - prompts: {prompts_array}")

        # Call Roboflow API
        print(f"🚀 Calling Roboflow API with text prompts...")
        print(f"   URL: {roboflow_url}")
        print(f"   Prompts (array): {prompts_array}")
        
        roboflow_response = requests.post(
            roboflow_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60  # 60 second timeout
        )
        
        print(f"📥 Roboflow response status: {roboflow_response.status_code}")

        if not roboflow_response.ok:
            error_details = roboflow_response.text
            print(f"❌ ROBOFLOW ERROR:")
            print(f"   Status: {roboflow_response.status_code}")
            print(f"   Details: {error_details}")
            
            return jsonify({
                'error': f'Roboflow API error: {roboflow_response.status_code}',
                'details': error_details,
                'status_code': roboflow_response.status_code,
                'message': 'Workflow configuration error. Check that SAM 3 has text prompt input enabled.'
            }), 500

        roboflow_data = roboflow_response.json()
        print("✓ Roboflow response received successfully")
        
        # DEBUG: Print full response to see structure
        import json
        print("\n" + "="*50)
        print("FULL ROBOFLOW RESPONSE:")
        print(json.dumps(roboflow_data, indent=2))
        print("="*50 + "\n")
        
        # CRITICAL DEBUG: Log what we're actually getting
        print("\n" + "🔍 DEBUGGING OUTPUT STRUCTURE:")
        if outputs and len(outputs) > 0:
            print(f"   Available keys: {list(outputs[0].keys())}")
            for key in outputs[0].keys():
                value = outputs[0][key]
                value_type = type(value).__name__
                if isinstance(value, dict):
                    print(f"   - {key} ({value_type}): {list(value.keys())}")
                elif isinstance(value, str):
                    print(f"   - {key} ({value_type}): {len(value)} chars")
                else:
                    print(f"   - {key} ({value_type})")
        print("="*50 + "\n")

        # Extract outputs
        outputs = roboflow_data.get('outputs', [])
        if not outputs:
            print("❌ No outputs in Roboflow response")
            return jsonify({'error': 'No outputs from Roboflow'}), 500

        print(f"📦 Number of outputs: {len(outputs)}")
        print(f"📦 Output keys: {list(outputs[0].keys())}")

        # Get SAM 3 predictions (segmentation results)
        all_predictions = []
        
        # Check for SAM 3 output
        if 'sam_3' in outputs[0]:
            sam_output = outputs[0]['sam_3']
            if isinstance(sam_output, dict):
                # SAM 3 might return predictions or masks
                all_predictions = sam_output.get('predictions', [])
                if not all_predictions:
                    all_predictions = sam_output.get('masks', [])
                print(f"📊 SAM 3 predictions found")
        
        # Alternative key names for SAM output
        elif 'sam3' in outputs[0]:
            sam_output = outputs[0]['sam3']
            if isinstance(sam_output, dict):
                all_predictions = sam_output.get('predictions', sam_output.get('masks', []))
        
        # Check for any segmentation or prediction keys
        else:
            for key in outputs[0].keys():
                if 'sam' in key.lower() or 'segment' in key.lower() or 'prediction' in key.lower():
                    pred_data = outputs[0][key]
                    if isinstance(pred_data, dict):
                        all_predictions = pred_data.get('predictions', pred_data.get('masks', []))
                        print(f"📊 Predictions found in key: {key}")
                        break
                    elif isinstance(pred_data, list):
                        all_predictions = pred_data
                        print(f"📊 Predictions found in key: {key}")
                        break
        
        print(f"📊 Total detections from SAM 3: {len(all_predictions)}")
        
        if all_predictions:
            print(f"📊 Sample prediction: {all_predictions[0]}")
        else:
            print("⚠️ WARNING: No predictions in response!")
            print(f"⚠️ Available keys in output: {list(outputs[0].keys())}")

        # Get visualization (annotated image with masks)
        annotated_image_base64 = None
        
        # Check for mask_visualization (most likely in your workflow)
        if 'mask_visualization' in outputs[0]:
            viz = outputs[0]['mask_visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz
            print(f"📊 Using mask_visualization")
        
        # Alternative visualization keys
        elif 'visualization' in outputs[0]:
            viz = outputs[0]['visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz
            print(f"📊 Using visualization")
        
        # Check any key with 'visual' or 'image'
        else:
            for key in outputs[0].keys():
                if 'visual' in key.lower() or 'annotated' in key.lower():
                    viz = outputs[0][key]
                    if isinstance(viz, dict):
                        annotated_image_base64 = viz.get('value', '')
                    elif isinstance(viz, str):
                        annotated_image_base64 = viz
                    print(f"📊 Using visualization from key: {key}")
                    break

        # Clean up base64 data URL if present
        if annotated_image_base64 and annotated_image_base64.startswith('data:'):
            annotated_image_base64 = annotated_image_base64.split(',', 1)[1]

        print(f"🎨 Visualization: {'✓ Found' if annotated_image_base64 else '❌ Not found'}")

        # CRITICAL: If no visualization, we need to handle this
        if not annotated_image_base64:
            print("⚠️  WARNING: No visualization image found!")
            print("⚠️  This might mean:")
            print("   1. Mask Visualization block is not connected")
            print("   2. SAM 3 found no objects to segment")
            print("   3. Workflow configuration issue")
            print(f"   4. Available output keys: {list(outputs[0].keys())}")
            
            # Try to provide helpful error message
            if len(all_predictions) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No objects detected',
                    'message': f"SAM 3 could not find any '{prompt}' in the image",
                    'prompt': prompt,
                    'total_detections': 0,
                    'detections': [],
                    'annotated_image': None
                }), 200  # Return 200 but with no detections
            else:
                return jsonify({
                    'success': False,
                    'error': 'Visualization not generated',
                    'message': 'Workflow returned predictions but no visualization image. Check Mask Visualization block.',
                    'prompt': prompt,
                    'total_detections': len(all_predictions),
                    'detections': [],
                    'annotated_image': None
                }), 200

        # Prepare response message
        if all_predictions:
            if len(all_predictions) == 1:
                message = f"Found 1 '{prompt}'"
            else:
                message = f"Found {len(all_predictions)} '{prompt}' objects"
        else:
            message = f"No '{prompt}' found in the image"

        response_data = {
            'success': True,
            'prompt': prompt,
            'annotated_image': annotated_image_base64,
            'detections': [
                {
                    'class': pred.get('class', prompt),  # Use prompt as class if not provided
                    'confidence': pred.get('confidence', 1.0),
                    'x': pred.get('x', 0),
                    'y': pred.get('y', 0),
                    'width': pred.get('width', 0),
                    'height': pred.get('height', 0),
                    'mask': pred.get('mask')  # Include mask data if available
                }
                for pred in all_predictions
            ],
            'total_detections': len(all_predictions),
            'message': message
        }

        print(f"✓ Response prepared: {message}")
        print("="*50 + "\n")
        return jsonify(response_data)

    except requests.Timeout:
        print("❌ Roboflow API timeout")
        return jsonify({
            'error': 'Request timeout',
            'details': 'The request took too long. Please try again.'
        }), 504
        
    except requests.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Network error',
            'details': str(e)
        }), 500

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

    finally:
        # Clean up uploaded file
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"🗑️  Cleaned up: {uploaded_file_path}")
            except Exception as e:
                print(f"⚠️  Failed to delete: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Finder AI Backend on port {port}")
    print(f"{'='*50}\n")
    app.run(debug=False, host='0.0.0.0', port=port)