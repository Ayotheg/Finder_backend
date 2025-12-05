# finder_backend.py
import os
import io
import json
import base64
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Load environment variables
load_dotenv()

# --- App config ---
app = Flask(__name__, static_folder="public")
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# --- Roboflow config ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "the-bunker")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "sam3-with-prompts")
ROBOFLOW_BASE = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_WORKFLOW_URL = f"{ROBOFLOW_BASE}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"

# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_image(file_path: str, quality: int = 85, max_dim: int = 1920) -> bytes:
    """Open, resize if large, convert to JPEG and return bytes."""
    try:
        img = Image.open(file_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()
    except Exception:
        with open(file_path, "rb") as f:
            return f.read()

# --- Routes ---
@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    uploaded_file_path = None
    try:
        # Validate file
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        # Prompt handling
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            prompt = "all objects"  # Default fallback

        # Save file temporarily
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(uploaded_file_path)

        # Compress & encode
        compressed_bytes = compress_image(uploaded_file_path)
        encoded_image = base64.b64encode(compressed_bytes).decode("utf-8")

        # Build payload for Roboflow SAM3 workflow
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": encoded_image,
                "text_prompt": prompt  # Must never be None
            },
        }

        # Call Roboflow workflow
        roboflow_resp = requests.post(
            ROBOFLOW_WORKFLOW_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if not roboflow_resp.ok:
            try:
                err_json = roboflow_resp.json()
            except Exception:
                err_json = roboflow_resp.text
            return jsonify({
                "error": f"Roboflow API error {roboflow_resp.status_code}",
                "details": err_json
            }), 500

        roboflow_data = roboflow_resp.json()
        outputs = roboflow_data.get("outputs", [])

        # Extract predictions and annotated image
        all_predictions = []
        annotated_image_base64 = None

        for out_block in outputs:
            if isinstance(out_block, dict):
                # SAM predictions
                for key in ["sam3", "sam", "predictions", "masks"]:
                    if key in out_block and isinstance(out_block[key], dict):
                        preds = out_block[key].get("predictions") or out_block[key].get("masks")
                        if preds:
                            all_predictions.extend(preds)

                # Mask visualization
                for vk in ["mask_visualization", "annotated_image", "visualization", "image"]:
                    if vk in out_block:
                        val = out_block[vk]
                        candidate = val.get("value") if isinstance(val, dict) else val
                        if isinstance(candidate, str) and candidate.strip():
                            annotated_image_base64 = candidate
                            if annotated_image_base64.startswith("data:"):
                                annotated_image_base64 = annotated_image_base64.split(",", 1)[1]
                            break
                if annotated_image_base64:
                    break

        # Normalize predictions
        normalized_detections = []
        for pred in all_predictions:
            if not isinstance(pred, dict):
                continue
            normalized_detections.append({
                "class": pred.get("class") or pred.get("label") or prompt,
                "confidence": pred.get("confidence", pred.get("score", 1.0)),
                "x": pred.get("x", 0),
                "y": pred.get("y", 0),
                "width": pred.get("width", pred.get("w", 0)),
                "height": pred.get("height", pred.get("h", 0)),
                "mask": pred.get("mask") or pred.get("encoded_mask") or pred.get("rle"),
            })

        # Build response
        response_data = {
            "success": True,
            "prompt": prompt,
            "annotated_image": annotated_image_base64,
            "detections": normalized_detections,
            "total_detections": len(normalized_detections),
            "message": f"Found {len(normalized_detections)} '{prompt}' objects" if normalized_detections else f"No '{prompt}' found"
        }

        return jsonify(response_data)

    except requests.Timeout:
        return jsonify({"error": "Request timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": "Network error", "details": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Finder AI Backend on port {port}")
    print(f"{'='*50}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
