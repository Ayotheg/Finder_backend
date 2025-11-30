from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urljoin
import traceback


load_dotenv()

app = Flask(__name__, static_folder='public')
CORS(app)

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

# Create uploads folder if it doesn't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        
        # Any prompt keyword contains the class name (e.g., "brush" matches "toothbrush")
        elif any(class_name in keyword for keyword in prompt_keywords):
            match = True
        
        if match:
            filtered.append(pred)
    
    return filtered if filtered else predictions  # Return all if no matches


FRONTEND_URL = os.getenv("FRONTEND_URL") 

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
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400

        # Get prompt from form data
        prompt = request.form.get('prompt', '').strip()

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)

        print(f"Processing image: {filename}")
        print(f"User prompt: '{prompt}'")

        # Read and encode image as base64
        with open(uploaded_file_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

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
        print("Calling Roboflow API...")
        roboflow_response = requests.post(
            roboflow_url,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if not roboflow_response.ok:
            error_msg = f"Roboflow API error: {roboflow_response.status_code}"
            print(error_msg)
            return jsonify({'error': error_msg, 'details': roboflow_response.text}), 500

        roboflow_data = roboflow_response.json()
        print("Roboflow response received successfully")

        # Extract predictions and visualization
        outputs = roboflow_data.get('outputs', [])
        if not outputs:
            return jsonify({'error': 'No outputs from Roboflow'}), 500

        # Get predictions
        predictions_data = outputs[0].get('predictions', {})
        all_predictions = predictions_data.get('predictions', [])

        # Get visualization - handle multiple formats
        annotated_image_base64 = None

        # Try getting from visualization field
        if 'visualization' in outputs[0]:
            viz = outputs[0]['visualization']
            if isinstance(viz, dict):
                annotated_image_base64 = viz.get('value', '')
            elif isinstance(viz, str):
                annotated_image_base64 = viz

        # If no visualization, try getting annotated_image directly
        if not annotated_image_base64 and 'annotated_image' in outputs[0]:
            annotated_image_base64 = outputs[0]['annotated_image']

        # Clean up the base64 string if it has a data URI prefix
        if annotated_image_base64 and annotated_image_base64.startswith('data:'):
            annotated_image_base64 = annotated_image_base64.split(',', 1)[1]

        print(f"Visualization found: {bool(annotated_image_base64)}")
        if annotated_image_base64:
            print(f"Visualization length: {len(annotated_image_base64)} characters")

        # Filter detections based on prompt
        filtered_detections = filter_detections_by_prompt(all_predictions, prompt)

        # Prepare response with helpful messages
        if not filtered_detections and prompt:
            # No matches found for the prompt
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

        # Return the response from the try block
        return jsonify(response_data)

    except Exception as e:
        # Log and return an error response
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

    finally:
        # Clean up uploaded file
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"Cleaned up temporary file: {uploaded_file_path}")
            except Exception as e:
                print(f"Failed to delete temporary file: {e}")


if __name__ == '__main__':
    # Use environment variable for port, default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)