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
    
    # If no prompt, return everything with 'all' status
    if not prompt or prompt.strip() == '':
        return predictions, "all"
    
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
    
    # Return filtered if found, otherwise all with 'no_match' status
    if filtered:
        return filtered, "matched"
    else:
        return predictions, "no_match"  # Show all but flag no match


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
        
        # DEBUG: Print full response to see structure
        import json
        print("\n" + "="*50)
        print("FULL ROBOFLOW RESPONSE:")
        print(json.dumps(roboflow_data, indent=2))
        print("="*50 + "\n")

        # Extract predictions and visualization
        outputs = roboflow_data.get('outputs', [])
        if not outputs:
            print("❌ No outputs in Roboflow response")
            return jsonify({'error': 'No outputs from Roboflow'}), 500

        print(f"📦 Number of outputs: {len(outputs)}")
        print(f"📦 Output keys: {outputs[0].keys() if outputs else 'None'}")

        # Get predictions - handle different response structures
        all_predictions = []
        
        # Method 1: Check for model_1 (your workflow structure)
        if 'model_1' in outputs[0]:
            model_output = outputs[0]['model_1']
            if isinstance(model_output, dict) and 'predictions' in model_output:
                all_predictions = model_output['predictions']
                print(f"📊 Predictions found in model_1")
        
        # Method 2: Check for direct predictions key
        elif 'predictions' in outputs[0]:
            predictions_data = outputs[0]['predictions']
            
            # Check if it's nested
            if isinstance(predictions_data, dict):
                all_predictions = predictions_data.get('predictions', [])
                print(f"📊 Predictions found in nested structure")
            elif isinstance(predictions_data, list):
                all_predictions = predictions_data
                print(f"📊 Predictions found in list structure")
        
        # Method 3: Check for object_detection_model output
        elif 'object_detection_model' in outputs[0]:
            obj_detection = outputs[0]['object_detection_model']
            if isinstance(obj_detection, dict):
                all_predictions = obj_detection.get('predictions', [])
                print(f"📊 Predictions found in object_detection_model")
        
        # Method 4: Check for any key containing 'model' or 'prediction'
        else:
            for key in outputs[0].keys():
                if 'model' in key.lower() or 'prediction' in key.lower():
                    pred_data = outputs[0][key]
                    if isinstance(pred_data, dict) and 'predictions' in pred_data:
                        all_predictions = pred_data['predictions']
                        print(f"📊 Predictions found in key: {key}")
                        break
                    elif isinstance(pred_data, list):
                        all_predictions = pred_data
                        print(f"📊 Predictions found in key: {key}")
                        break
        
        print(f"📊 Total detections from Roboflow: {len(all_predictions)}")
        
        if all_predictions:
            print(f"📊 Sample prediction: {all_predictions[0]}")
        else:
            print("⚠️ WARNING: No predictions in response!")
            print(f"⚠️ Available keys in output: {list(outputs[0].keys())}")

        # Get visualization - handle different response structures
        annotated_image_base64 = None
        
        # Method 1: Check for 'visualization' key
        if 'visualization' in outputs[0]:
            viz = outputs[0]['visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz
        
        # Method 2: Check for 'bounding_box_visualization' key
        elif 'bounding_box_visualization' in outputs[0]:
            viz = outputs[0]['bounding_box_visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz
            print(f"📊 Using bounding_box_visualization")
        
        # Method 3: Check for 'annotated_image' key
        elif 'annotated_image' in outputs[0]:
            annotated_image_base64 = outputs[0]['annotated_image']
        
        # Method 4: Check any key with 'visual' or 'image'
        else:
            for key in outputs[0].keys():
                if 'visual' in key.lower() or 'image' in key.lower():
                    viz = outputs[0][key]
                    if isinstance(viz, dict):
                        annotated_image_base64 = viz.get('value', '')
                    elif isinstance(viz, str):
                        annotated_image_base64 = viz
                    print(f"📊 Using visualization from key: {key}")
                    break

        if annotated_image_base64 and annotated_image_base64.startswith('data:'):
            annotated_image_base64 = annotated_image_base64.split(',', 1)[1]

        print(f"🎨 Visualization: {'✓ Found' if annotated_image_base64 else '❌ Not found'}")

        # Apply smart filtering
        filtered_detections, filter_status = filter_detections_by_prompt(all_predictions, prompt)
        print(f"🔍 Filter status: {filter_status}")
        print(f"🔍 Filtered detections: {len(filtered_detections)}")

        # Get list of detected classes for better messaging
        detected_classes = list(set(p.get('class', 'unknown') for p in all_predictions))

        # Prepare response based on filter status
        if filter_status == "no_match" and prompt:
            # Prompt was used but nothing matched - show all anyway
            message = f"'{prompt}' not found. Showing detected: {', '.join(detected_classes)}"
        elif filter_status == "matched":
            # Found what user was looking for
            message = f"Found {len(filtered_detections)} {prompt}(s)"
        else:  # filter_status == "all"
            # No prompt - showing everything
            if detected_classes:
                message = f"Found {len(filtered_detections)} item(s): {', '.join(detected_classes)}"
            else:
                message = "No objects detected in image"

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
            'filter_status': filter_status,
            'detected_classes': detected_classes,
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