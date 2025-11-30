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
    'workflow_id': 'find-brushes-books-vr-headsets-slippers-charger-heads-toast-machines-bags-and-changeovers'
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


def filter_detections_by_prompt(predictions, prompt):
    """Filter detections based on user prompt with improved fuzzy matching"""
    if not prompt:
        return predictions
    
    prompt_lower = prompt.lower().strip()
    filtered = []
    
    # Split prompt into keywords
    prompt_keywords = prompt_lower.split()
    
    for pred in predictions:
        class_name = pred.get('class', '').lower()
        
        # Check various matching strategies
        match = False
        
        # Exact match
        if class_name == prompt_lower:
            match = True
        
        # Class name contains any prompt keyword
        elif any(keyword in class_name for keyword in prompt_keywords):
            match = True
        
        # Any prompt keyword contains the class name
        elif any(class_name in keyword for keyword in prompt_keywords):
            match = True
        
        if match:
            filtered.append(pred)
    
    return filtered if filtered else predictions


@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        'name': 'Finder AI Backend',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'analyze': '/api/analyze'
        }
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
    """Main endpoint for image analysis"""
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

        # Prepare Roboflow API request
        roboflow_url = f"{ROBOFLOW_CONFIG['api_url']}/{ROBOFLOW_CONFIG['workspace']}/workflows/{ROBOFLOW_CONFIG['workflow_id']}"
        
        payload = {
            'api_key': ROBOFLOW_CONFIG['api_key'],
            'inputs': {
                'image': {
                    'type': 'base64',
                    'value': encoded_image
                }
            }
        }

        # Call Roboflow API
        print(f"🚀 Calling Roboflow API...")
        print(f"   URL: {roboflow_url}")
        
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
                'message': 'Workflow configuration error. Please check your Roboflow workflow.'
            }), 500

        roboflow_data = roboflow_response.json()
        print("✓ Roboflow response received successfully")

        # Extract predictions and visualization
        outputs = roboflow_data.get('outputs', [])
        if not outputs:
            print("❌ No outputs in Roboflow response")
            return jsonify({'error': 'No outputs from Roboflow'}), 500

        # Get predictions
        predictions_data = outputs[0].get('predictions', {})
        all_predictions = predictions_data.get('predictions', [])
        print(f"📊 Total detections: {len(all_predictions)}")

        # Get visualization
        annotated_image_base64 = None
        
        if 'visualization' in outputs[0]:
            viz = outputs[0]['visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz

        if not annotated_image_base64 and 'annotated_image' in outputs[0]:
            annotated_image_base64 = outputs[0]['annotated_image']

        if annotated_image_base64 and annotated_image_base64.startswith('data:'):
            annotated_image_base64 = annotated_image_base64.split(',', 1)[1]

        print(f"🎨 Visualization: {'✓ Found' if annotated_image_base64 else '❌ Not found'}")

        # Filter detections based on prompt
        filtered_detections = filter_detections_by_prompt(all_predictions, prompt)
        print(f"🔍 Filtered detections: {len(filtered_detections)}")

        # Prepare response
        if not filtered_detections and prompt:
            detected_classes = list(set(p.get('class', 'unknown') for p in all_predictions))
            response_data = {
                'success': True,
                'prompt': prompt,
                'annotated_image': annotated_image_base64,
                'detections': [],
                'total_detections': len(all_predictions),
                'filtered_detections': 0,
                'message': f"No '{prompt}' found. Detected: {', '.join(detected_classes)}"
            }
        else:
            response_data = {
                'success': True,
                'prompt': prompt,
                'annotated_image': annotated_image_base64,
                'detections': [
                    {
                        'class': pred.get('class'),
                        'confidence': pred.get('confidence'),
                        'x': pred.get('x'),
                        'y': pred.get('y'),
                        'width': pred.get('width'),
                        'height': pred.get('height')
                    }
                    for pred in filtered_detections
                ],
                'total_detections': len(all_predictions),
                'filtered_detections': len(filtered_detections),
                'message': f"Found {len(filtered_detections)} {prompt}(s)" if prompt else f"Found {len(filtered_detections)} item(s)"
            }

        print("✓ Response prepared successfully")
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